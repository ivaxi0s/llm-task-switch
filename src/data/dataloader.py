import random

from src import SEED
from tqdm import tqdm
from copy import deepcopy
from typing import List, Dict, Tuple, Any, Generator
from datasets import load_dataset, DatasetDict
from dataclasses import dataclass
from abc import abstractmethod
import numpy as np


def load_data(data_name: str, lim: int = None) -> Tuple["train", "val", "test"]:
    data_ret = {
        "rt": _load_rotten_tomatoes,
        "gigaword": _load_gigaword,
    }
    return data_ret[data_name](lim)


class PromptLoader:
    """Class to load prompts from different datasets

    Each dataset is immediately loaded into cache
    """

    def __init__(self, incontext: str, eval: str):
        """Load all the datasets into memory"""

        if eval == "gigaword":
            self.eval_set = GigawordDataLoader()
        if incontext == "gigaword":
            self.incontext_set = GigawordDataLoader()

    def load_prompt(self, num_examples: int):
        """Return prompts from different datasets
        prompt = incontext + eval

        The prompts are pre-loaded into memory.
        This is because not much RAM is required,
        and the pre-processing is slow.

        % TODO: add functionality to limit the test size(deterministically)

        % TODO: if this is implemened as a dataset transform,
        then this can be cached (tho this will only save around 40s)

        Args:
            incontext: name of the dataset to load incontext prompts from
            eval: name of the dataset to load evaluation prompts from
            num_examples: number of incontext examples to include
        """

        prompts = [
            (
                self.incontext_set.incontext_prompt(num_examples, seed=idx)
                + "\n"
                + eval_prompt
            )
            for idx, eval_prompt in enumerate(self.eval_set.eval_prompt())
        ]

        return prompts

    def load_testdata(self) -> list[str]:
        """Return the test data reference as a list[str]

        This is used for evaluation
        """
        return self.eval_set.load_test_reference()


@dataclass
class DataLoader:
    """Abstract class for loading data"""

    dataset_name: str
    _dataset: DatasetDict | None = None

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = load_dataset(self.dataset_name)
        return self._dataset

    @abstractmethod
    def incontext_prompt(self, num_examples: int, seed: int = SEED):
        ...

    @abstractmethod
    def eval_prompt(self):
        ...


# @dataclass
# class RottenTomatoesDataLoader(DataLoader):


class GigawordDataLoader(DataLoader):
    """DataLoader for gigaword dataset

    NOTE: DatasetDict is of the form:
    {train, validation, test} with features: {document, summary}
    """

    PROMPT_PREFIX = "Please read the following pairs of texts and summaries:\n"

    def __init__(self):
        super().__init__(dataset_name="gigaword")

        # Map the training set to incontext prompts
        self.train = self.dataset["train"]
        self.train = self.train.map(GigawordDataLoader._prompt)

        # Map the test set to evaluation prompts
        self.test = self.dataset["test"]
        self.test = self.test.map(GigawordDataLoader._eval_prompt)

    def load_test_reference(self):
        """Return the test data as a list[str]"""
        return self.test["summary"]

    @staticmethod
    def _prompt(example: dict[str, Any]) -> dict[str, Any]:
        """Transform a single example to incontext prompt"""
        return {
            "prompt": (
                "article: " + example["document"] + "\nsummary: " + example["summary"]
            ),
        }

    @staticmethod
    def _eval_prompt(example: dict[str, Any]) -> dict[str, Any]:
        """Transform a single example to evaluation prompt"""
        return {
            "eval_prompt": "Please summarize the following article.\n"
            + example["document"],
        }

    def incontext_prompt(self, num_examples: int, seed: int = SEED):
        """Returns prompt for incontext examples

        Args:
            num_examples: number of incontext examples to include
            seed: random seed for selecting examples. e.g. this could be the iteration number
        """
        if num_examples == 0:
            return ""
        out = GigawordDataLoader.PROMPT_PREFIX
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(self.train), num_examples, replace=False)
        examples = self.train.select(idxs, keep_in_memory=True)["prompt"]

        # examples = self.train.shuffle(seed=seed, keep_in_memory=True).select(
        #     range(num_examples), keep_in_memory=True
        # )["prompt"]
        out = out + "\n".join(examples)
        return out

    def eval_prompt(self) -> Generator[str, None, None]:
        """Yields prompt for evaluation examples"""

        for eval_prompt in self.test["eval_prompt"]:
            yield eval_prompt


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
