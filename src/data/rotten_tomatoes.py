from dataclasses import dataclass
from src.data.dataloader import DataLoader
from src import SEED
from typing import Any
import numpy as np


@dataclass
class RottenTomatoesDataLoader(DataLoader):
    """Dataloader for rotten tomatoes dataset

    NOTE: DatasetDict is of the form:
    {train, validation, test} with features: {text, label}

    The `labels` are mapped to `sentiments`
    1 -> positive; 0 -> negative
    """

    PROMPT_PREFIX = (
        "Please read the following pairs of movie reviews and sentiment:\n\n"
    )
    EVAL_PROMPT_PREFIX = (
        # "Please perform a Sentiment Classification task. "
        # "Given the following movie review, assign a sentiment label from ['negative', 'positive'].\n"
        "Can you choose only one sentiment ['negative', 'positive'] for this review.\n"
    )
    EVAL_PROMPT_SUFFIX = (
        ""
        "\nReturn only the sentiment label without any other text."
        # "\nChoose ONLY one from the ['negative', 'positive']. "
        # "\nThink step by step."
        " Make sure to follow the format otherwise your answer will be disqualified:\n"
        "<Answer> positive / negative </Answer>.\n Do not output neutral."
    )

    def __init__(self):
        super().__init__(dataset_path="rotten_tomatoes")

        # Map all labels to sentiments
        self._dataset = self.dataset.map(RottenTomatoesDataLoader._label_to_sentiment)
        self._dataset = self._dataset.map(
            RottenTomatoesDataLoader._add_answer_tags, load_from_cache_file=True
        )

        # Map the training set to incontext prompts
        self.train = self.dataset["train"]
        self.train = self.train.map(
            RottenTomatoesDataLoader._prompt, load_from_cache_file=True
        )

        # Map the test set to evaluation prompts
        self.test = self.dataset["test"]
        self.test = self.test.map(
            RottenTomatoesDataLoader._eval_prompt, load_from_cache_file=True
        )

    def load_test_reference(self):
        """Return the test data as a list[str]"""
        return self.test["sentiment"]

    def load_likelihood_reference(self):
        """Return the reference data for likelihoods calculation"""
        return self.test["target_with_tags"]

    @staticmethod
    def _add_answer_tags(example: dict[str, Any]) -> dict[str, str]:
        """Add answer tags to the target"""
        return {"target_with_tags": "<Answer> " + example["sentiment"] + " </Answer>"}

    @staticmethod
    def _label_to_sentiment(example: dict[str, Any]) -> dict[str, str]:
        """Map the label to sentiment"""
        return {
            "sentiment": "positive" if example["label"] == 1 else "negative",
        }

    @staticmethod
    def _prompt(example: dict[str, Any]) -> dict[str, str]:
        """Transform a single example to incontext prompt"""
        return {
            "prompt": (
                "review: "
                + example["text"]
                + "\nsentiment: "
                + example["target_with_tags"]
            ),
        }

    @staticmethod
    def _eval_prompt(example: dict[str, Any]) -> dict[str, str]:
        """Transform a single example to evaluation prompt"""
        return {
            "eval_prompt": (
                RottenTomatoesDataLoader.EVAL_PROMPT_PREFIX
                + "review: "
                + example["text"]
                + RottenTomatoesDataLoader.EVAL_PROMPT_SUFFIX
            ),
        }

    def incontext_prompt(self, num_examples: int, seed: int = SEED):
        """Returns prompt for incontext examples

        Args:
            num_examples: number of incontext examples to include
            seed: random seed for selecting examples. e.g. this could be the iteration number
        """
        if num_examples == 0:
            return ""
        out = RottenTomatoesDataLoader.PROMPT_PREFIX
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(self.train), num_examples, replace=False)
        examples = self.train.select(idxs, keep_in_memory=True)["prompt"]

        out = out + "\n".join(examples) + "\n\n"
        return out

    def incontext_prompt_iterative(self, num_examples: int, seed: int = SEED):
        """Returns prompt for incontext examples

        Args:
            num_examples: number of incontext examples to include
            seed: random seed for selecting examples. e.g. this could be the iteration number
        """

        out: list[dict] = []
        if num_examples == 0:
            return out
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(self.train), num_examples, replace=False)
        examples = self.train.select(idxs, keep_in_memory=True)["prompt"]

        for ex in examples:
            parts = ex.split("\nsentiment: ")
            out.append(
                {
                    "role": "user",
                    "content": RottenTomatoesDataLoader.EVAL_PROMPT_PREFIX
                    + parts[0]
                    + RottenTomatoesDataLoader.EVAL_PROMPT_SUFFIX,
                }
            )
            out.append({"role": "assistant", "content": parts[1]})
        return out
