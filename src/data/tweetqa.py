from dataclasses import dataclass
from src.data.dataloader import DataLoader
from src import SEED
from typing import Any
import numpy as np


class TweetQADataLoader(DataLoader):
    """DataLoader for TweetQA dataset"""

    PROMPT_PREFIX = "Please read the following triplet of contexts, questions and answers and summaries:\n\n"

    def __init__(self):
        super().__init__(dataset_path="lmqg/qag_tweetqa")

        # Map the training set to incontext prompts
        self.train = self.dataset["train"]
        self.train = self.train.map(TweetQADataLoader._prompt)

        # Map the test set to evaluation prompts
        self.test = self.dataset["test"]
        self.test = self.test.map(TweetQADataLoader._eval_prompt)

    def load_test_reference(self):
        """Return the test data as a list[str]"""
        return [a[0] for a in self.test["answers"]]

    @staticmethod
    def _prompt(example: dict[str, Any]) -> dict[str, str]:
        """Transform a single example to incontext prompt"""
        return {
            "prompt": (
                "tweet: "
                + example["paragraph"]
                + "\nquestion: "
                + example["questions"][0]
                + "\nanswer: "
                + example["answers"][0]
            ),
        }

    @staticmethod
    def _eval_prompt(example: dict[str, Any]) -> dict[str, str]:
        """Transform a single example to evaluation prompt"""
        return {
            "eval_prompt": "Read the given tweet and answer the corresponding question.\n"
            "tweet: " + example["paragraph"] + "\nquestion: " + example["questions"][0]
        }

    def incontext_prompt(self, num_examples: int, seed: int = SEED):
        """Returns prompt for incontext examples

        Args:
            num_examples: number of incontext examples to include
            seed: random seed for selecting examples. e.g. this could be the iteration number
        """
        if num_examples == 0:
            return ""
        out = TweetQADataLoader.PROMPT_PREFIX
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(self.train), num_examples, replace=False)

        examples = self.train.select(idxs, keep_in_memory=True)["prompt"]

        # examples = self.train.shuffle(seed=seed, keep_in_memory=True).select(
        #     range(num_examples), keep_in_memory=True
        # )["prompt"]
        out = out + "\n".join(examples) + "\n\n"
        return out

    def incontext_prompt_iterative(self, num_examples: int, seed: int = SEED):
        """Returns prompt for incontext examples

        Args:
            num_examples: number of incontext examples to include
            seed: random seed for selecting examples. e.g. this could be the iteration number
        """
        if num_examples == 0:
            return []

        out = []
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(self.train), num_examples, replace=False)
        examples = self.train.select(idxs, keep_in_memory=True)["prompt"]

        # examples = self.train.shuffle(seed=seed, keep_in_memory=True).select(
        #     range(num_examples), keep_in_memory=True
        # )["prompt"]

        # out = out + "\n".join(examples) + "\n"
        for ex in examples:
            command = "Read the given tweet and answer the corresponding question.\n"
            parts = ex.split("\nanswer: ")
            out.append({"role": "user", "content": command + parts[0]})
            out.append({"role": "assistant", "content": parts[1]})
        return out
