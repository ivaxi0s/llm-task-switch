import random

from src import SEED
from tqdm import tqdm
from typing import Any, Generator
from datasets import load_dataset, DatasetDict, concatenate_datasets
from dataclasses import dataclass
from abc import abstractmethod
import numpy as np
import re
from transformers import AutoTokenizer


class PromptLoader:
    """Class to load prompts from different datasets

    Each dataset is immediately loaded into cache
    """

    def __init__(self, incontext: str, eval: str):
        """Load all the datasets into memory"""

        if eval == "gigaword":
            self.eval_set = GigawordDataLoader()
        elif eval == "dailymail":
            self.eval_set = DailymailDataLoader()
        elif eval == "rotten_tomatoes":
            self.eval_set = RottenTomatoesDataLoader()
        elif eval == "tweetqa":
            self.eval_set = TweetQADataLoader()
        elif eval == "gsm8k":
            self.eval_set = GSM8KDataLoader()
        elif eval == "mmluaa":
            self.eval_set = MMLUAbstractAlgebraDataLoader()
        else:
            raise ValueError(f"Unknown eval dataset: {eval}")

        if incontext == eval:
            # Prevent having to loading same dataset twice
            self.incontext_set = self.eval_set
        elif incontext == "gigaword":
            self.incontext_set = GigawordDataLoader()
        elif incontext == "dailymail":
            self.incontext_set = DailymailDataLoader()
        elif incontext == "wikicat":
            self.incontext_set = WikicatDataLoader()
        elif incontext == "rotten_tomatoes":
            self.incontext_set = RottenTomatoesDataLoader()
        elif incontext == "tweetqa":
            self.incontext_set = TweetQADataLoader()
        elif incontext == "gsm8k":
            self.incontext_set = GSM8KDataLoader()
        elif incontext == "mmluaa":
            self.incontext_set = MMLUAbstractAlgebraDataLoader()
        else:
            raise ValueError(f"Unknown incontext dataset: {incontext}")

    def load_prompt(self, num_examples: int, eval_size: int):
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
                idx,
                self.incontext_set.incontext_prompt(num_examples, seed=idx)
                + eval_prompt,
            )
            for idx, eval_prompt in self.eval_set.eval_prompt(eval_size)
        ]
        eval_idxs, prompts = zip(*prompts)

        return eval_idxs, prompts

    def load_prompt_iterative(self, num_examples: int, eval_size: int):
        """Return prompts from different datasets - iterative version of prompts: returns list of lists of dictionary
        first list iterates through samples in test dataset
        second list iterates through the user/assistant messages in turn
        the dictionary has keys
             'role': which is either 'user' or 'assistant;
             'content': the message in that turn

        e.g. the list of message for a single sample with a single incontent example will be
                [
            {'role': 'user',
            'content': 'you are a summary system'.\n What is the summary of (1)
            },

            {'role': 'assistannt',
            'content' : summary of (1)
            }

                {'role': user,
                'content': What is the summary of eval_sample
            }
                ]
        """

        # prompts = [
        #     (self.incontext_set.incontext_prompt(num_examples, seed=idx) + eval_prompt)
        #     for idx, eval_prompt in enumerate(self.eval_set.eval_prompt())
        # ]

        idxs_prompts = [
            (
                idx,  # idx of the sample in the test dataset
                self.incontext_set.incontext_prompt_iterative(num_examples, seed=idx)
                + [{"role": "user", "content": eval_prompt}],
            )
            for idx, eval_prompt in self.eval_set.eval_prompt(eval_size)
        ]
        eval_idxs, prompts = zip(*idxs_prompts)

        return eval_idxs, prompts

    def load_testdata(self, eval_idxs) -> list[str]:
        """Return the test data reference as a list[str]

        This is used for evaluation
        """
        references = self.eval_set.load_test_reference()
        if eval_idxs is None:
            return references
        else:
            return [references[idx] for idx in eval_idxs]

    def load_likelihood_reference(self, eval_idxs) -> list[str]:
        """Return the reference data for likelihoods calculation"""
        references = self.eval_set.load_likelihood_reference()
        if eval_idxs is None:
            return references
        else:
            return [references[idx] for idx in eval_idxs]


@dataclass
class DataLoader:
    """Abstract class for loading data"""

    dataset_path: str
    dataset_name: str | None = None
    _dataset: DatasetDict | None = None
    test: DatasetDict | None = None

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = load_dataset(self.dataset_path, self.dataset_name)

        return self._dataset

    @abstractmethod
    def incontext_prompt(self, num_examples: int, seed: int = SEED):
        ...

    @abstractmethod
    def load_test_reference(self):
        ...

    def load_likelihood_reference(self):
        """Return the reference data for likelihoods calculation"""
        return self.load_test_reference()

    def eval_prompt(
        self, eval_size: int, seed: int = SEED
    ) -> Generator[str, None, None]:
        """Yields prompt for evaluation examples

        We sample eval_size examples from the test set
        """

        if eval_size is None:
            eval_examples = self.test["eval_prompt"]
        elif eval_size > 0:
            if eval_size > len(self.test):
                print(
                    f"WARNING: eval_size ({eval_size}) is larger than test set size ({len(self.test)})"
                )
                eval_size = len(self.test)
            rng = np.random.default_rng(seed)
            idxs = rng.choice(len(self.test), eval_size, replace=False)
            eval_examples = self.test.select(idxs, keep_in_memory=True)["eval_prompt"]
        else:
            raise ValueError(f"eval_size must be None or > 0, got {eval_size}")

        for idx, eval_prompt in enumerate(eval_examples):
            yield idx, eval_prompt


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
            RottenTomatoesDataLoader._add_answer_tags, load_from_cache_file=False
        )

        # Map the training set to incontext prompts
        self.train = self.dataset["train"]
        self.train = self.train.map(
            RottenTomatoesDataLoader._prompt, load_from_cache_file=False
        )

        # Map the test set to evaluation prompts
        self.test = self.dataset["test"]
        self.test = self.test.map(
            RottenTomatoesDataLoader._eval_prompt, load_from_cache_file=False
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

        out = []
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


class DailymailDataLoader(DataLoader):
    """DataLoader for dailymail dataset

    NOTE: DatasetDict is of the form:
    {train, validation, test} with features: {'article', 'highlights', 'id'}
    """

    PROMPT_PREFIX = "Please read the following pairs of texts and summaries:\n\n"

    def __init__(self):
        super().__init__(dataset_path="cnn_dailymail", dataset_name="3.0.0")

        # Map the training set to incontext prompts
        self.train = self.dataset["train"]
        self.train = self.train.map(DailymailDataLoader._prompt)
        print("Removing large training set examples")
        print("Original training set size: ", len(self.train))
        self.train = self._remove_large_training_set_examples()
        print("New Training set size: ", len(self.train))

        # Map the test set to evaluation prompts
        self.test = self.dataset["test"]
        self.test = self.test.map(DailymailDataLoader._eval_prompt)

    def load_test_reference(self):
        """Return the test data reference (answers) as a list[str]"""
        return self.test["highlights"]

    def _remove_large_training_set_examples(self, max_prompt_len: int = 1792):
        """Remove examples with large token size from training set

        This is so that the in context examples aren't so big
        """
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        return self.train.filter(
            lambda example: (
                len(tokenizer(example["prompt"])["input_ids"]) < max_prompt_len
            )
        )

    @staticmethod
    def _prompt(example: dict[str, Any]) -> dict[str, str]:
        """Transform a single example to incontext prompt"""
        return {
            "prompt": (
                "article: " + example["article"] + "\nsummary: " + example["highlights"]
            ),
        }

    @staticmethod
    def _eval_prompt(example: dict[str, Any]) -> dict[str, str]:
        """Transform a single example to evaluation prompt"""
        return {
            "eval_prompt": "Please summarize the following article.\n"
            + example["article"],
        }

    def incontext_prompt(self, num_examples: int, seed: int = SEED):
        """Returns prompt for incontext examples

        Args:
            num_examples: number of incontext examples to include
            seed: random seed for selecting examples. e.g. this could be the iteration number
        """
        if num_examples == 0:
            return ""
        out = DailymailDataLoader.PROMPT_PREFIX
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
        if num_examples == 0:
            return []
        out = []
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(self.train), num_examples, replace=False)
        examples = self.train.select(idxs, keep_in_memory=True)["prompt"]

        for ex in examples:
            command = "Please summarize the following article.\n"
            parts = ex.split("\nsummary: ")
            out.append({"role": "user", "content": command + parts[0]})
            out.append({"role": "assistant", "content": parts[1]})
        return out


class WikicatDataLoader(DataLoader):
    """DataLoader for wikicat dataset

    NOTE: DatasetDict is of the form:
    {train, validation, test} with features: {document, summary}
    """

    PROMPT_PREFIX = "Please read the following pairs of texts and summaries:\n\n"

    def __init__(self):
        super().__init__(dataset_path="GEM/wiki_cat_sum")

        # Map the training set to incontext prompts
        self.train = self.dataset["train"]
        self.train = self.train.map(WikicatDataLoader._prompt)

        # Map the test set to evaluation prompts
        self.test = self.dataset["test"]
        self.test = self.test.map(WikicatDataLoader._eval_prompt)

    def load_test_reference(self):
        """Return the test data as a list[str]"""
        return self.test["summary"]

    @staticmethod
    def _prompt(example: dict[str, Any]) -> dict[str, str]:
        """Transform a single example to incontext prompt"""
        return {
            "prompt": (
                "article: "
                + " ".join(example["paragraphs"])
                + "\nsummary: "
                + " ".join(example["summary"]["text"])
            ),
        }

    @staticmethod
    def _eval_prompt(example: dict[str, Any]) -> dict[str, str]:
        """Transform a single example to evaluation prompt"""
        return {
            "eval_prompt": "Please summarize the following article.\n"
            + " ".join(example["paragraphs"]),
        }

    def incontext_prompt(self, num_examples: int, seed: int = SEED):
        """Returns prompt for incontext examples

        Args:
            num_examples: number of incontext examples to include
            seed: random seed for selecting examples. e.g. this could be the iteration number
        """
        if num_examples == 0:
            return ""
        out = WikicatDataLoader.PROMPT_PREFIX
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
            command = "Please summarize the following article.\n"
            parts = ex.split("\nsummary: ")
            out.append({"role": "user", "content": command + parts[0]})
            out.append({"role": "assistant", "content": parts[1]})
        return out


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

    def __init__(self):
        super().__init__(dataset_path="lukaemon/mmlu", dataset_name="abstract_algebra")

        # Map the training set to incontext prompts
        self.train = concatenate_datasets(
            [self.dataset["train"], self.dataset["validation"]]
        )
        self.train = self.train.map(
            MMLUAbstractAlgebraDataLoader._add_answer_tags, load_from_cache_file=False
        )
        self.train = self.train.map(
            MMLUAbstractAlgebraDataLoader._prompt, load_from_cache_file=False
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
