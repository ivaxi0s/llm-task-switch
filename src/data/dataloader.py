import random

from tqdm import tqdm
from copy import deepcopy
from typing import List, Dict, Tuple
from datasets import load_dataset
from dataclasses import dataclass
from abc import abstractmethod


def load_data(data_name: str, lim: int = None) -> Tuple["train", "val", "test"]:
    data_ret = {
        "rt": _load_rotten_tomatoes,
        "gigaword": _load_gigaword,
    }
    return data_ret[data_name](lim)


@dataclass
class DataLoader:
    """Abstract class for loading data"""

    dataset_name: str

    def load_data(self):
        dataset = load_dataset(self.dataset_name)

    @abstractmethod
    def incontext_prompt_template(self):
        ...

    @abstractmethod
    def eval_prompt_template(self):
        ...


# @dataclass
# class RottenTomatoesDataLoader(DataLoader):


def _load_rotten_tomatoes(lim: int = None):
    dataset = load_dataset("rotten_tomatoes")
    train = list(dataset["train"])[:lim]
    val = list(dataset["validation"])[:lim]
    test = list(dataset["test"])[:lim]

    # Modify the keys based on the template tags (see the paper)
    train = [change_key(t, "text", "Review") for t in train]
    val = [change_key(t, "text", "Review") for t in val]
    test = [change_key(t, "text", "Review") for t in test]

    train = [change_key(t, "label", "Sentiment") for t in train]
    val = [change_key(t, "label", "Sentiment") for t in val]
    test = [change_key(t, "label", "Sentiment") for t in test]

    mapping = {0: "negative", 1: "positive"}
    train = [content_map(t, "Sentiment", mapping) for t in train]
    val = [content_map(t, "Sentiment", mapping) for t in val]
    test = [content_map(t, "Sentiment", mapping) for t in test]
    return train, val, test


def _load_gigaword(lim: int = None, test_valid_lim: int = None):
    dataset = load_dataset("gigaword")
    train = list(dataset["train"])[:test_valid_lim]
    val = list(dataset["validation"])[:test_valid_lim]
    test = list(dataset["test"])[:lim]

    train = [change_key(t, "document", "Text") for t in train]
    val = [change_key(t, "document", "Text") for t in val]
    test = [change_key(t, "document", "Text") for t in test]

    train = [change_key(t, "summary", "Summary") for t in train]
    val = [change_key(t, "summary", "Summary") for t in val]
    test = [change_key(t, "summary", "Summary") for t in test]

    # Save train, val, test datasets for easy re-use
    

    return train, val, test


def _create_splits(examples: list, ratio=0.8) -> Tuple[list, list]:
    examples = deepcopy(examples)
    split_len = int(ratio * len(examples))

    random.seed(1)
    random.shuffle(examples)

    split_1 = examples[:split_len]
    split_2 = examples[split_len:]
    return split_1, split_2


def change_key(ex: dict, old_key="content", new_key="text"):
    """convert key name from the old_key to 'text'"""
    ex = ex.copy()
    ex[new_key] = ex.pop(old_key)
    return ex


def content_map(ex: dict, target_key, mapping):
    ex[target_key] = mapping[ex[target_key]]
    return ex


def _multi_key_to_text(ex: dict, key1: str, key2: str):
    """concatenate contents of key1 and key2 and map to name text"""
    ex = ex.copy()
    ex["text"] = ex.pop(key1) + " " + ex.pop(key2)
    return ex


def _invert_labels(ex: dict):
    ex = ex.copy()
    ex["label"] = 1 - ex["label"]
    return ex


def _map_labels(ex: dict, map_dict={-1: 0, 1: 1}):
    ex = ex.copy()
    ex["label"] = map_dict[ex["label"]]
    return ex


def _rand_sample(lst, frac):
    random.Random(4).shuffle(lst)
    return lst[: int(len(lst) * frac)]
