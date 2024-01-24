import json
import re

import evaluate as hf_evl


def evaluate(pred_fpath, ref_data, ref_data_name):
    """
    pred_fpath: contains model predictions as a list (of dict)
    ref_data: test data with reference labels
    ref_data_name: specifies the reference dataset name (influences evaluation metric)
    """
    # load predictions from cache
    with open(pred_fpath, "r") as f:
        pred_data = json.load(f)

    dataset_eval = {
        "rt": eval_rt,
        "gigaword": eval_gigaword,
        "dailymail": eval_dailymail,
        "tweetqa": eval_tweetqa,
    }

    # evaluate predictions
    if ref_data_name in dataset_eval:
        return dataset_eval[ref_data_name](pred_data, ref_data)
    else:
        raise ValueError(f"Unknown dataset: {ref_data_name}")


def eval_rt(pred_data: list[str], ref_data: list[str]):
    """Evaluate the accuracy of sentiment predictions

    Evaluation metric(s) for Rotten Tomatoes dataset
    https://huggingface.co/datasets/rotten_tomatoes

    Potentially: update how to measure accuracy in case of failure to identify label
    - Either we take into account more cases
    - or we perhaps for emit the tokens that we're looking for (e.g. "sentiment: positive")
    """
    assert len(pred_data) == len(ref_data)
    matches = 0
    failed = 0
    for pred, ref in zip(pred_data, ref_data):
        pred = pred.lower()
        # pred_sent = re.findall("<sentiment>(.*?)</sentiment>", pred, re.DOTALL)
        # pred_sent = re.findall("^sentiment:(.*?)$", pred, re.DOTALL)
        pred = pred.strip()
        # if len(pred_sent) != 1:
        #     failed += 1
        #     continue
        # pred_sent = pred_sent[0]
        # pred_sent = pred_sent.strip(" ")
        if pred == ref:
            matches += 1
        elif not pred in ("positive", "negative"):
            failed += 1
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

    return scores


def eval_dailymail(pred_data, ref_data):
    """Evaluate dailymail dataset using ROUGE metric"""
    assert len(pred_data) == len(ref_data)

    rouge = hf_evl.load("rouge")

    # rouge = load_metric("rouge")
    scores = rouge.compute(predictions=pred_data, references=ref_data)

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
