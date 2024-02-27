from dataclasses import dataclass
from src.data.dataloader import DataLoader
from src import SEED
from typing import Any
import numpy as np


class GigawordDataLoader(DataLoader):
    """DataLoader for gigaword dataset

    NOTE: DatasetDict is of the form:
    {train, validation, test} with features: {document, summary}
    """

    PROMPT_PREFIX = "Please read the following pairs of texts and summaries:\n\n"

    def __init__(self):
        super().__init__(dataset_path="gigaword")

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
    def _prompt(example: dict[str, Any]) -> dict[str, str]:
        """Transform a single example to incontext prompt"""
        return {
            "prompt": (
                "article: " + example["document"] + "\nsummary: " + example["summary"]
            ),
        }

    @staticmethod
    def _eval_prompt(example: dict[str, Any]) -> dict[str, str]:
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
        # out = GigawordDataLoader.PROMPT_PREFIX
        out = []
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(self.train), num_examples, replace=False)
        examples = self.train.select(idxs, keep_in_memory=True)["prompt"]

        # examples = self.train.shuffle(seed=seed, keep_in_memory=True).select(
        #     range(num_examples), keep_in_memory=True
        # )["prompt"]

        # out = out + "\n".join(examples) + "\n"
        for ex in examples:
            command = "Please summarize the following article.\n"
            parts = ex.split("\nsummary: ")
            out.append({"role": "user", "content": command + parts[0]})
            out.append({"role": "assistant", "content": parts[1]})
        return out
