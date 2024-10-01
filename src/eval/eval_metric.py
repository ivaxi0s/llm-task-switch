import json
import re
from pathlib import Path
import evaluate as hf_evl  # type: ignore
import numpy as np
import pickle
from typing import Callable
from transformers import AutoTokenizer  # type: ignore


# RE patterns for extracting config information from results folder path
RE_INCTXT = re.compile(r"^incontext_data_(.*)")  # Extract in context set
RE_NUM_EXAMPLES = re.compile(r"^num_examples_(.*)")  # Extract number of examples
RE_EVAL_DATA = re.compile(r"^eval_data_(.*)")  # Extract eval data name
RE_ANSWER_TAGS = re.compile(r"<Answer>(.*?)</Answer>")

llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")


def get_dataset_eval_func(dataset: str) -> Callable:
    dataset_eval = {
        "rotten_tomatoes": eval_rt,
        "gigaword": eval_gigaword,
        "dailymail": eval_dailymail,
        "tweetqa": eval_tweetqa,
        "gsm8k": eval_gsm8k,
        "mmluaa": eval_mmluaa,
        "moral": eval_mmluaa,
        "mmlu-math": eval_mmluaa,
    }
    if "mmlu" in dataset:
        dataset_eval[dataset] = eval_mmluaa

    # evaluate predictions
    if not dataset in dataset_eval:
        raise ValueError(f"Unknown dataset: {dataset}")

    return dataset_eval[dataset]


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
        f = pred_fpath.parent
        for _ in range(6):
            f = f.parent
            m = RE_EVAL_DATA.match(f.name)
            if m is not None:
                ref_data_name = m.group(1)
                break
        else:
            raise ValueError(f"Could not extract eval data name from {pred_fpath}")
        # m = RE_EVAL_DATA.match(pred_fpath.parent.parent.parent.parent.name)
        # if m is None:
        #     raise ValueError(f"Could not extract eval data name from {pred_fpath}")
        # ref_data_name = m.group(1)

    # load predictions from cache
    with open(pred_fpath, "r") as f:
        pred_data = json.load(f)

    # Evaluate predictions
    results = get_dataset_eval_func(ref_data_name)(pred_data, ref_data)
    # include config information in results
    results |= extract_config_from_path(pred_fpath)
    # Calculate baseline likelihood
    results |= calculate_likelihood(pred_fpath, ref_data_name)

    # Calculate token lengths
    results |= calculate_token_size(pred_fpath, results["model"])

    # Store results in json in same dir as pred_fpath
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    return results


def evaluate_converse(
    pred_fpath: Path,
    ref_data: list,
    ref_data_name: str | None = None,
    use_cached=False,
) -> dict:
    """Calculate results for predictions in the style of a conversation

    pred_fpath:

    Assuming predictions is in the format:
        [
            [zero shot responses],
            [response after 1st turn],
            ...
        ]
    """
    # Check if results are already cached
    results_file = Path(pred_fpath).parent / "results.json"
    if use_cached and results_file.is_file():
        # load results from cache
        with open(results_file, "r") as f:
            return json.load(f)

    if ref_data_name is None:
        m = RE_EVAL_DATA.match(pred_fpath.parent.parent.parent.parent.parent.name)
        if m is None:
            raise ValueError(f"Could not extract eval data name from {pred_fpath}")
        ref_data_name = m.group(1)

    # load predictions from cache
    with open(pred_fpath, "r") as f:
        predictions = json.load(f)

    # include config information in results
    # breakpoint()
    results = extract_config_from_path(pred_fpath.parent.parent)
    # Calculate baseline likelihood
    for idx, p in enumerate(predictions):
        assert len(p) == len(ref_data)
        results |= {
            f"history_length_{idx}": get_dataset_eval_func(ref_data_name)(p, ref_data)
        }

    # Evaluate predictions
    # results = get_dataset_eval_func(ref_data_name)(predictions, ref_data)
    # results |= calculate_likelihood(pred_fpath, ref_data_name)
    # Store results in json in same dir as pred_fpath
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    return results


def calculate_token_size(pred_fpath: Path, model: str):
    """Return the mean token size of prompts, predictions and full conversation

    Args:
        pred_fpath: ...dir/predictions.json
    """
    tokens = {}
    if model == "llama-7b":
        tokenizer = llama_tokenizer
    elif model == "mistral-7b":
        tokenizer = mistral_tokenizer
    else:
        return tokens

    # load predictions
    with open(pred_fpath, "r") as f:
        pred_data = json.load(f)
        pred_tokens_length = tokenizer(
            pred_data,
            return_attention_mask=False,
            return_length=True,
        )["length"]
        pred_tokens_length = np.mean(pred_tokens_length)

    # load prompts
    prompt_fpath = pred_fpath.parent / "prompts.json"
    with open(prompt_fpath, "r") as f:
        prompt_data = json.load(f)
        # check if conversation history length is 0
        if len(prompt_data[0]) == 1:
            conversation_history_length = 0
        else:
            conversation_history_length = [
                len(tokenizer.apply_chat_template(prompt[:-1]))
                for prompt in prompt_data
            ]
            conversation_history_length = np.mean(conversation_history_length)
        # prompt_tokens_length = [
        #     len(tokenizer.apply_chat_template(prompt)) for prompt in prompt_data
        # ]
        # prompt_tokens_length = np.mean(prompt_tokens_length)
        target_task_length = [
            len(tokenizer.apply_chat_template(prompt[-1:])) for prompt in prompt_data
        ]
        target_task_length = np.mean(target_task_length)

    tokens |= {
        # "mean_prompt_tokens_length": prompt_tokens_length,
        "mean_conversation_history_length": conversation_history_length,
        "mean_target_task_length": target_task_length,
        "mean_pred_tokens_length": pred_tokens_length,
        "mean_conversation_length": conversation_history_length
        + target_task_length
        + pred_tokens_length,
    }

    return tokens


def calculate_likelihood(pred_fpath: Path, ref_data_name: str):
    """Calculate the Expected likelihood of the baseline response"""
    fpaths = {
        "base_likelihood": pred_fpath.parent / "base_probabilities.pkl",
        "final_likelihood": pred_fpath.parent / "final_probabilities.pkl",
        "ref_likelihood": pred_fpath.parent / "ref_probabilities.pkl",
    }

    # fpath = Path(pred_fpath.parent / "base_probabilities.pkl")
    results: dict = {}

    # Check which model this is
    model_name = pred_fpath.parent.parent.parent.parent.parent.name

    for key, fpath in fpaths.items():
        if not fpath.is_file():
            print(f"Reference probabilities not found at {fpath}")
            results[key] = None
            continue
        ref_probs = pickle.load(open(fpath, "rb"))

        # Remove the first-token probability in llama, because it is spurious
        if model_name == "llama-7b":
            # remove the first token prob
            ref_probs = [probs[1:] for probs in ref_probs]

        # Likelihood is product over all tokens
        # log_likelihood = [float(np.sum(np.log(probs))) for probs in ref_probs]

        # Likelihood is mean over all tokens
        log_likelihood = [float(np.mean(np.log(probs))) for probs in ref_probs]

        results[key] = log_likelihood
    return results


def extract_config_from_path(results_folder: str | Path, subdir="iterative"):
    """Extract config information from results folder path

    The folder path name is of the following form
    experiments/
        <model>/eval_data_<eval_set>/incontext_data<incontext_data>
            num_examples_<#>/iterative/
                                /seed_<#>/
    """
    config = {}
    results_folder = Path(results_folder)
    if results_folder.is_file():
        results_folder = results_folder.parent
    if "seed" in results_folder.name:
        config |= {"seed": int(results_folder.name.split("_")[-1])}
        results_folder = results_folder.parent
    if results_folder.name == subdir:
        results_folder = results_folder.parent

    config |= {
        "model": results_folder.parent.parent.parent.name,
        "incontext_set": RE_INCTXT.match(results_folder.parent.name).group(1),
        "num_examples": int(RE_NUM_EXAMPLES.match(results_folder.name).group(1)),
    }

    return config


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


def eval_gigaword(pred_data, ref_data):
    """Evaluate gigaword dataset using ROUGE metric"""
    assert len(pred_data) == len(ref_data)

    rouge = hf_evl.load("rouge")

    scores = rouge.compute(predictions=pred_data, references=ref_data)
    scores |= {"mean_num_of_chars": mean_num_of_characters(pred_data)}

    return scores


def eval_dailymail(pred_data, ref_data):
    """Evaluate dailymail dataset using ROUGE metric"""
    assert len(pred_data) == len(ref_data)

    rouge = hf_evl.load("rouge")

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


def eval_gsm8k(pred_data: list, ref_data: list):
    """Evaluate gms8k dataset using accuracy"""
    assert len(pred_data) == len(ref_data)
    matches = 0
    failed = 0
    for pred, ref in zip(pred_data, ref_data):
        # Extract the number from the end of the string
        # Assume <Answer> number </Answer>
        pred_answers = RE_ANSWER_TAGS.findall(pred)
        if not pred_answers:
            failed += 1
            continue
        answer = pred_answers[-1]
        answer = answer.strip()
        # Remove commas and space from answer
        answer = answer.replace(",", "")
        answer = answer.replace(" ", "")

        try:
            answer = float(answer)
            if answer.is_integer():
                answer = int(answer)
            else:
                continue
            if answer == ref:
                matches += 1
        except Exception as e:
            failed += 1
            print(
                f"---Failed for Answer: {answer}\n: Reference: {ref}\n: Exception: {e}"
            )
            continue
    return {
        "matches": matches,
        "failed": failed,
        "total": len(pred_data),
    }


def eval_mmluaa(pred_data: list, ref_data: list):
    """Evaluate mmluaa dataset using accuracy"""
    assert len(pred_data) == len(ref_data)
    matches = 0
    failed = 0
    for pred, ref_letters_texts in zip(pred_data, ref_data):
        ref, *ref_text = ref_letters_texts
        # Extract the number from the end of the string
        # Assume <Answer> letter </Answer>
        pred_answers = RE_ANSWER_TAGS.findall(pred)
        if not pred_answers:
            failed += 1
            continue
        # If there are multuple answer tags, then we count this as a fail
        if len(pred_answers) > 1:
            failed += 1
            print(f"---Failed; Multiple Answer: {pred_answers}\n: Reference: {ref}")
            continue

        answer = pred_answers[0]
        answer = answer.strip()

        if len(answer) == 1:
            if answer.upper() == ref:
                matches += 1
            continue

        # Otherwise, perhaps the text is in the answer tags
        # Convert reference and answers to lowercase
        ref_text = [text.lower() for text in ref_text]
        answer = answer.lower()

        correct_ref_text_idx = ord(ref) - ord("A")
        correct_text = ref_text.pop(correct_ref_text_idx)
        # Check if the correct text is in the prediction
        if correct_text in answer:
            matches += 1
            continue
        # Check if any of the other texts are in the prediction
        if any(text in answer for text in ref_text):
            continue
        #  If none of the other texts are in the prediction,
        #  then we failed to extract the model prediction
        failed += 1
        print(
            f"---Failed for Answer: {answer}\n: Reference: {ref}, {ref_text}:\nPrediction: {pred}"
        )

    return {
        "matches": matches,
        "failed": failed,
        "total": len(pred_data),
    }


def mean_num_of_characters(pred_data):
    """Average number of characters of the predictions"""
    return sum(len(pred) for pred in pred_data) / len(pred_data)
