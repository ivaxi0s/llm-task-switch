from datasets import DatasetDict, concatenate_datasets  # type: ignore
from src.data.dataloader import DataLoader
from src import SEED
from typing import Any
import numpy as np


class MMLUDataLoader(DataLoader):
    """Abstract class for MMLU datasets

    NOTE: DatasetDict is of the form:
    {train, validation, test} with features: {input, A,B,C,D,target}
    """

    def __init__(self, dataset_name: str, prompt_prefix: str):
        super().__init__(dataset_path="lukaemon/mmlu", dataset_name=dataset_name)

        fn_kwargs = {"prompt_prefix": prompt_prefix}

        # Map the answers to include tags
        self._dataset = self.dataset.map(
            MMLUDataLoader._add_answer_tags, load_from_cache_file=True
        )
        # Map the test set to prompts
        self._dataset = self.dataset.map(
            MMLUDataLoader._eval_prompt, load_from_cache_file=True, fn_kwargs=fn_kwargs
        )

        # Join train and validation sets
        self.train: DatasetDict = concatenate_datasets(
            [self.dataset["train"], self.dataset["validation"]]
        )
        self.train = DataLoader.remove_large_dataset_examples(
            self.train, column="eval_prompt"
        )

        self.test = self.dataset["test"]
        self.test = self.test.map(
            MMLUDataLoader._target_text, load_from_cache_file=True
        )

    @staticmethod
    def _add_answer_tags(example: dict[str, Any]) -> dict[str, str]:
        """Add answer tags to the target"""
        return {"target_with_tags": "<Answer> " + example["target"] + " </Answer>"}

    @staticmethod
    def _target_text(example: dict[str, Any]) -> dict[str, tuple]:
        """Create a column `target_text` with tuple(target, *answers)

        Column contains the tuple(
            target, Answer A, Answer B, Answer C, Answer D
        )
        """
        letter = example["target"]
        return {
            "target_text": (
                letter.upper(),
                example["A"],
                example["B"],
                example["C"],
                example["D"],
            )
        }

    def load_test_reference(self):
        return self.test["target_text"]

    def load_likelihood_reference(self):
        """Return the test data as a list[str] to be used for likelihood calculation"""
        return self.test["target_with_tags"]

    @staticmethod
    def _eval_prompt(example: dict[str, Any], prompt_prefix: str) -> dict[str, str]:
        """Transform a single example to evaluation prompt"""
        return {
            "eval_prompt": (
                prompt_prefix
                + "\nQuestion: "
                + example["input"]
                + "\n(A) "
                + example["A"]
                + "\n(B) "
                + example["B"]
                + "\n(C) "
                + example["C"]
                + "\n(D) "
                + example["D"]
            ),
        }

    def incontext_prompt_iterative(self, num_examples: int, seed: int = SEED):
        """Returns prompt for incontext examples

        Args:
            num_examples: number of incontext
        """
        if num_examples == 0:
            return []

        out = []
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(self.train), num_examples, replace=False)
        examples = self.train.select(idxs, keep_in_memory=True)
        prompts = examples["eval_prompt"]
        answers = examples["target_with_tags"]

        for prompt, answer in zip(prompts, answers):
            out.append({"role": "user", "content": prompt})
            out.append({"role": "assistant", "content": answer})
        return out


class MMLUElementaryMathematicsDataLoader(MMLUDataLoader):
    """DataLoader for MMLU Elementary Mathematics dataset"""

    PROMPT_PREFIX = (
        "You have a multiple choice question on Elementary Mathematics. "
        "Only one of the options is correct: A, B, C, or D. "
        "Give your answer in the following format with the tags provided: "
        "<Answer> </Answer>. "
        "Please read the following question and options and answer the question.\n"
    )

    def __init__(self, dataset_name="elementary_mathematics"):
        super().__init__(dataset_name=dataset_name, prompt_prefix=self.PROMPT_PREFIX)


class MMLUHumanAgingDataLoader(MMLUDataLoader):
    """DataLoader for MMLU Human Aging dataset"""

    PROMPT_PREFIX = (
        "You have a multiple choice question on Human Aging. "
        "Only one of the options is correct: A, B, C, or D. "
        "Give your answer in the following format with the tags provided: "
        "<Answer> </Answer>. "
        "Please read the following question and options and answer the question.\n"
    )

    def __init__(self, dataset_name="human_aging"):
        super().__init__(dataset_name=dataset_name, prompt_prefix=self.PROMPT_PREFIX)


class MMLUProfessionalLawDataLoader(MMLUDataLoader):
    """DataLoader for MMLU Professional Law dataset"""

    PROMPT_PREFIX = (
        "You have a multiple choice question on Professional Law. "
        "Only one of the options is correct: A, B, C, or D. "
        "Give your answer in the following format with the tags provided: "
        "<Answer> </Answer>. "
        "Please read the following question and options and answer the question.\n"
    )

    def __init__(self, dataset_name="professional_law"):
        super().__init__(dataset_name=dataset_name, prompt_prefix=self.PROMPT_PREFIX)


class MMLUAbstractAlgebraDataLoader(DataLoader):
    """DataLoader for MMLU Abstract Algebra dataset

    NOTE: DatasetDict is multiple choice of the form:
    {train, validation, test} with features: {input, A,B,C,D,target}
    """

    PROMPT_PREFIX = (
        "You have a multiple choice question on Abstract Algebra. "
        "Only one of the options is correct: A, B, C, or D. "
        # "If you think the answer is A, then say <Answer> A </Answer>. "
        # "If you think the answer is B, then say <Answer> B </Answer>. "
        # "If you think the answer is C, then say <Answer> C </Answer>. "
        # "If you think the answer is D, then say <Answer> D </Answer>. "
        "Give your answer in the following format with the tags provided: "
        "<Answer> </Answer>. "
        # "where option is one of A, B, C, D.\n"
        "Please read the following question and options and answer the question.\n"
    )
    PROMPT_SUFFIX = ""

    def __init__(self, dataset_name="abstract_algebra"):
        super().__init__(dataset_path="lukaemon/mmlu", dataset_name=dataset_name)

        # Map the training set to incontext prompts
        self.train = concatenate_datasets(
            [self.dataset["train"], self.dataset["validation"]]
        )
        self.train = self.train.map(
            MMLUAbstractAlgebraDataLoader._add_answer_tags, load_from_cache_file=True
        )
        self.train = self.train.map(
            MMLUAbstractAlgebraDataLoader._prompt, load_from_cache_file=True
        )

        # Map the test set to evaluation prompts
        self.test = self.dataset["test"]
        self.test = self.test.map(
            MMLUAbstractAlgebraDataLoader._eval_prompt, load_from_cache_file=False
        )
        self.test = self.test.map(
            MMLUAbstractAlgebraDataLoader._target_text, load_from_cache_file=False
        )
        self.test = self.test.map(
            MMLUAbstractAlgebraDataLoader._add_answer_tags, load_from_cache_file=False
        )

    @staticmethod
    def _add_answer_tags(example: dict[str, Any]) -> dict[str, str]:
        """Add answer tags to the target"""
        return {"target_with_tags": "<Answer> " + example["target"] + " </Answer>"}

    @staticmethod
    def _target_text(example: dict[str, Any]) -> dict[str, tuple]:
        """Column for the target answer as text"""
        letter = example["target"]
        return {
            "target_text": (
                letter.upper(),
                example["A"],
                example["B"],
                example["C"],
                example["D"],
            )
        }

    def load_test_reference(self):
        """Return the test data as a list[str]"""
        return self.test["target_text"]

    def load_likelihood_reference(self):
        """Return the test data as a list[str] to be used for likelihood calculation"""
        return self.test["target_with_tags"]

    @staticmethod
    def _prompt(example: dict[str, Any]) -> dict[str, str]:
        """Transform a single example to incontext prompt"""
        return {
            "prompt": (
                MMLUAbstractAlgebraDataLoader.PROMPT_PREFIX
                + "\nQuestion: "
                + example["input"]
                + "\n(A) "
                + example["A"]
                + "\n(B) "
                + example["B"]
                + "\n(C) "
                + example["C"]
                + "\n(D) "
                + example["D"]
                + "\nanswer: "
                + example["target_with_tags"]
            ),
        }

    @staticmethod
    def _eval_prompt(example: dict[str, Any]) -> dict[str, str]:
        """Transform a single example to evaluation prompt"""
        return {
            "eval_prompt": MMLUAbstractAlgebraDataLoader.PROMPT_PREFIX
            + "\nQuestion: "
            + example["input"]
            + " (A) "
            + example["A"]
            + " (B) "
            + example["B"]
            + " (C) "
            + example["C"]
            + " (D) "
            + example["D"]
        }

    def incontext_prompt_iterative(self, num_examples: int, seed: int = SEED):
        """Returns prompt for incontext examples

        Args:
            num_examples: number of incontext
        """
        if num_examples == 0:
            return []

        out = []
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(self.train), num_examples, replace=False)
        examples = self.train.select(idxs, keep_in_memory=True)["prompt"]

        for ex in examples:
            parts = ex.split("\nanswer: ")
            out.append({"role": "user", "content": parts[0]})
            out.append({"role": "assistant", "content": parts[1]})
        return out


class MMLUMoralScenariosDataLoader(MMLUAbstractAlgebraDataLoader):
    """Dataloader for MMLU Moral Scenarios dataset

    NOTE: DatasetDict is multiple choice of the form:
    {train, validation, test} with features: {input, A,B,C,D,target}
    """

    PROMPT_PREFIX = (
        "You have a multiple choice question on Moral Scenarios. "
        "Only one of the options is correct: A, B, C, or D. "
        # "If you think the answer is A, then say <Answer> A </Answer>. "
        # "If you think the answer is B, then say <Answer> B </Answer>. "
        # "If you think the answer is C, then say <Answer> C </Answer>. "
        # "If you think the answer is D, then say <Answer> D </Answer>. "
        "Give your answer in the following format with the tags provided: "
        "<Answer> </Answer>. "
        # "where option is one of A, B, C, D.\n"
        "Please read the following question and options and answer the question.\n"
    )

    def __init__(self):
        super().__init__(dataset_name="moral_scenarios")
