from src import SEED
from dataclasses import dataclass
from datasets import load_dataset, DatasetDict  # type: ignore
from transformers import AutoTokenizer  # type: ignore
from abc import abstractmethod
from typing import Generator


@dataclass
class DataLoader:
    """Abstract class for loading data"""

    dataset_path: str
    dataset_name: str | None = None
    _dataset: DatasetDict | None = None
    test: DatasetDict | None = None

    @property
    def dataset(self) -> DatasetDict:
        if self._dataset is None:
            self._dataset = load_dataset(self.dataset_path, self.dataset_name)

        return self._dataset

    @staticmethod
    def remove_large_dataset_examples(
        ds: DatasetDict, column: str, max_prompt_len: int = 1792
    ):
        """Remove examples with large token size from training set

        This is so that the in context examples aren't so big
        """
        print("Removing large training set examples")
        print("Original training set size: ", len(ds))
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        ds = ds.filter(
            lambda example: (
                len(tokenizer(example[column])["input_ids"]) < max_prompt_len
            )
        )
        print("New Training set size: ", len(ds))
        # print("Max token length: ", max(ds[column]))
        return ds

    @abstractmethod
    def load_test_reference(self):
        ...

    def load_likelihood_reference(self):
        """Return the reference data for likelihoods calculation"""
        return self.load_test_reference()

    def eval_prompt(
        self, eval_size: int | None, seed: int = SEED
    ) -> Generator[str, None, None]:
        """Yields prompt for evaluation examples

        We sample eval_size examples from the test set
        """

        if eval_size is None:
            eval_examples = self.test["eval_prompt"]
            idxs = range(len(self.test))
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

        for idx, eval_prompt in zip(idxs, eval_examples):
            yield int(idx), eval_prompt
