from dataclasses import dataclass
from src.data.dataloader import DataLoader
from src import SEED
from typing import Any
import numpy as np
import re


class GSM8KDataLoader(DataLoader):
    """DataLoader for GSM8K dataset

    NOTE: DatasetDict is of the form:
    {train, validation, test} with features: {question, answer}
    """

    # Regex to extract "#### Number" from the answer
    r_number = re.compile(r"#### (.*)$")
    r_answer = re.compile(r"#### .*$")

    EVAL_PROMPT_PREFIX = (
        "Answer the following question. "
        "Think step by step. "
        "Give your final answer in the following format with the tags provided: "
        "<Answer> number </Answer>. "
        "Your answer must be a numerical integer and exclude units.\n"
    )
    EVAL_PROMPT_SUFFIX = ""

    def __init__(self):
        super().__init__(dataset_path="gsm8k", dataset_name="main")

        # Map the training set to add answer tags
        self._dataset = self.dataset.map(GSM8KDataLoader._add_answer_tags)

        # Map the training set to incontext prompts
        self.train = self.dataset["train"]
        self.train = self.train.map(GSM8KDataLoader._prompt, load_from_cache_file=False)

        # Map the test set to evaluation prompts
        self.test = self.dataset["test"]
        self.test = self.test.map(
            GSM8KDataLoader._eval_prompt, load_from_cache_file=False
        )

    def load_test_reference(self):
        """Return the test data as a list[str]"""
        return self.test["value"]

    @staticmethod
    def _add_answer_tags(example: dict[str, Any]) -> dict[str, str]:
        """Add answer tags to the example"""
        answer = example["answer"]
        # Extract the number from the answer
        vals = GSM8KDataLoader.r_number.findall(answer)
        if len(vals) != 1:
            raise ValueError("Value not found in ", answer)
        val = vals[0]
        # Remove comma
        val = val.replace(",", "")
        # Check if value is int
        val = float(val)
        if not val.is_integer():
            print(val)
            raise ValueError("Value is not an integer")
        val = int(val)

        answer = GSM8KDataLoader.r_answer.sub(f"<Answer> {val} </Answer>", answer)

        return {"answer_with_tags": answer, "value": val}

    @staticmethod
    def _prompt(example: dict[str, Any]) -> dict[str, str]:
        """Transform a single example to incontext prompt"""
        return {
            "prompt": (
                GSM8KDataLoader.EVAL_PROMPT_PREFIX
                + "question: "
                + example["question"]
                + GSM8KDataLoader.EVAL_PROMPT_SUFFIX
                + "\nanswer: "
                + example["answer_with_tags"]
            ),
        }

    @staticmethod
    def _eval_prompt(example: dict[str, Any]) -> dict[str, str]:
        """Transform a single example to evaluation prompt"""
        return {
            "eval_prompt": GSM8KDataLoader.EVAL_PROMPT_PREFIX
            + "question: "
            + example["question"]
            + GSM8KDataLoader.EVAL_PROMPT_SUFFIX
        }

    def incontext_prompt_iterative(self, num_examples: int, seed: int = SEED):
        """Returns prompt for incontext examples

        Args:
            num_examples: number of incontext examples to include
            seed: random seed for selecting examples. e.g. this could be the iteration number

        Returns:
            list of dictionaries with keys 'role' and 'content'
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
            parts = ex.split("\nanswer: ")
            out.append({"role": "user", "content": parts[0]})
            out.append({"role": "assistant", "content": parts[1]})
        return out
