import json
import re
from pathlib import Path
import evaluate as hf_evl

# RE patterns for extracting config information from results folder path
RE_INCTXT = re.compile(r"^incontext_data_(.*)")  # Extract in context set
RE_NUM_EXAMPLES = re.compile(r"^num_examples_(.*)")  # Extract number of examples
RE_EVAL_DATA = re.compile(r"^eval_data_(.*)")  # Extract eval data name


def evaluate(
    pred_fpath: Path,
    ref_data: list,
    ref_data_name: str | None = None,
    use_cached=False,
) -> dict:
    """
    pred_fpath: Path to predictions.json file
                containing model predictions as a list (of dict)
    ref_data: test data with reference labels
    ref_data_name: specifies the reference dataset name (influences evaluation metric)
    use_cached: use cached predictions if True
    """

    # Check if results are already cached
    results_file = Path(pred_fpath).parent / "results.json"
    if use_cached and results_file.is_file():
        # load results from cache
        with open(results_file, "r") as f:
            return json.load(f)

    if ref_data_name is None:
        ref_data_name = RE_EVAL_DATA.match(
            pred_fpath.parent.parent.parent.parent.name
        ).group(1)

    # load predictions from cache
    with open(pred_fpath, "r") as f:
        pred_data = json.load(f)

    dataset_eval = {
        "rotten_tomatoes": eval_rt,
        "gigaword": eval_gigaword,
        "dailymail": eval_dailymail,
        "tweetqa": eval_tweetqa,
    }

    # evaluate predictions
    if not ref_data_name in dataset_eval:
        raise ValueError(f"Unknown dataset: {ref_data_name}")

    # Evaluate predictions
    results = dataset_eval[ref_data_name](pred_data, ref_data)
    # include config information in results
    results |= extract_config_from_path(pred_fpath)
    # Store results in json in same dir as pred_fpath
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    return results


def extract_config_from_path(results_folder: str | Path, subdir="iterative"):
    """Extract config information from results folder path

    The folder path name is of the following form
    experiments/
        <model>/eval_data_<eval_set>/incontext_data<incontext_data>
            num_examples_<#>/iterative/
    """
    results_folder = Path(results_folder)
    if results_folder.is_file():
        results_folder = results_folder.parent
    if results_folder.name == subdir:
        results_folder = results_folder.parent
    return {
        "model": results_folder.parent.parent.parent.name,
        "incontext_set": RE_INCTXT.match(results_folder.parent.name).group(1),
        "num_examples": int(RE_NUM_EXAMPLES.match(results_folder.name).group(1)),
    }


def eval_rt(pred_data: list[str], ref_data: list[str]):
    """Evaluate the accuracy of sentiment predictions

    Evaluation metric(s) for Rotten Tomatoes dataset
    https://huggingface.co/datasets/rotten_tomatoes

    Potentially: update how to measure accuracy in case of failure to identify label
    - Either we take into account more cases
    - or we perhaps force emit the tokens that we're looking for (e.g. "sentiment: positive")
    """
    assert len(pred_data) == len(ref_data)
    matches = 0
    failed = 0
    for pred, ref in zip(pred_data, ref_data):
        pred = pred.lower()
        # pred = re.findall("<sentiment>(.*?)</sentiment>", pred, re.DOTALL)
        # pred = re.findall("<answer>(.*?)</answer>", pred, re.DOTALL)
        # pred_sent = re.findall("^sentiment:(.*?)$", pred, re.DOTALL)
        # if len(pred) != 1:
        #     failed += 1
        #     continue
        # pred_sent = pred_sent[0]
        # pred_sent = pred_sent.strip(" ")
        positive = "positive" in pred
        negative = "negative" in pred
        if positive and negative or (not positive and not negative):
            failed += 1
            print(f"Reference: {ref}:\nPrediction: {pred}")
            continue
        if positive and ref == "positive":
            matches += 1
        elif negative and ref == "negative":
            matches += 1
        # breakpoint()
        # pred = pred[0].strip()
        # if pred == ref:
        #     matches += 1
        # elif not pred in ("positive", "negative"):
        #     failed += 1
    print(
        f"Generation format failed for {failed/len(pred_data)*100:0f}% \
          ({failed}/{len(pred_data)} samples)"
    )
    return {
        "matches": matches,
        "failed": failed,
        "total": len(pred_data),
    }
    # return {
    #     "Accuracy": f"{matches/(len(pred_data)-failed)*100:.2f}% \
    #         ({matches}/{len(pred_data)-failed} samples)"
    # }


def eval_gigaword(pred_data, ref_data):
    """Evaluate gigaword dataset using ROUGE metric"""
    assert len(pred_data) == len(ref_data)

    rouge = hf_evl.load("rouge")

    # rouge = load_metric("rouge")
    scores = rouge.compute(predictions=pred_data, references=ref_data)
    scores |= {"mean_num_of_chars": mean_num_of_characters(pred_data)}

    return scores


def eval_dailymail(pred_data, ref_data):
    """Evaluate dailymail dataset using ROUGE metric"""
    assert len(pred_data) == len(ref_data)

    rouge = hf_evl.load("rouge")

    # rouge = load_metric("rouge")
    scores = rouge.compute(predictions=pred_data, references=ref_data)
    scores |= {"mean_num_of_chars": mean_num_of_characters(pred_data)}

    return scores


def eval_tweetqa(pred_data: list, ref_data: list):
    """Evaluate tweetqa dataset using ROUGE and METEOR metrics

    tweetqa dataset:
    https://huggingface.co/datasets/tweet_qa

    ROUGE:
    https://huggingface.co/spaces/evaluate-metric/rouge

    METEOR:
    https://huggingface.co/spaces/evaluate-metric/meteor
    """
    assert len(pred_data) == len(ref_data)

    rouge = hf_evl.load("rouge")
    meteor = hf_evl.load("meteor")
    rouge_scores: dict = rouge.compute(predictions=pred_data, references=ref_data)
    meteor_scores: dict = meteor.compute(predictions=pred_data, references=ref_data)

    return rouge_scores | meteor_scores


def mean_num_of_characters(pred_data):
    """Average number of characters of the predictions"""
    return sum(len(pred) for pred in pred_data) / len(pred_data)
