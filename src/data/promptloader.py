# import datasets
from src.data.dailymail import DailymailDataLoader
from src.data.gigaword import GigawordDataLoader
from src.data.rotten_tomatoes import RottenTomatoesDataLoader
from src.data.tweetqa import TweetQADataLoader
from src.data.gsm8k import GSM8KDataLoader
from src.data.mmlu import (
    MMLUAbstractAlgebraDataLoader,
    MMLUMoralScenariosDataLoader,
    MMLUElementaryMathematicsDataLoader,
    MMLUHumanAgingDataLoader,
    MMLUProfessionalLawDataLoader,
)


class PromptLoader:
    """Class to load prompts from different datasets

    Each dataset is immediately loaded into cache
    """

    def __init__(self, incontext: str, eval: str):
        """Load all the datasets into memory"""

        dataloader_dict: dict = {
            "gigaword": GigawordDataLoader,
            "dailymail": DailymailDataLoader,
            # "wikicat": WikicatDataLoader,
            "rotten_tomatoes": RottenTomatoesDataLoader,
            "tweetqa": TweetQADataLoader,
            "gsm8k": GSM8KDataLoader,
            "mmluaa": MMLUAbstractAlgebraDataLoader,
            "mmlu-moral": MMLUMoralScenariosDataLoader,
            "mmlu-math": MMLUElementaryMathematicsDataLoader,
            "mmlu-age": MMLUHumanAgingDataLoader,
            "mmlu-law": MMLUProfessionalLawDataLoader,
        }

        if not eval in dataloader_dict:
            raise ValueError(f"Unknown eval dataset: {eval}")
        if not incontext in dataloader_dict:
            raise ValueError(f"Unknown incontext dataset: {incontext}")

        self.eval_set = dataloader_dict[eval]()

        if incontext == eval:
            # Prevent having to loading same dataset twice
            self.incontext_set = self.eval_set
        else:
            self.incontext_set = dataloader_dict[incontext]()

    def load_prompt_iterative(
        self, num_examples: int, eval_size: int | None, seed_multiplier: int = 1
    ):
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
        if seed_multiplier == 0:
            raise ValueError("seed cannot be 0")

        # prompts = [
        #     (self.incontext_set.incontext_prompt(num_examples, seed=idx) + eval_prompt)
        #     for idx, eval_prompt in enumerate(self.eval_set.eval_prompt())
        # ]

        idxs_prompts = [
            (
                idx,  # idx of the sample in the test dataset
                self.incontext_set.incontext_prompt_iterative(
                    num_examples, seed=idx * seed_multiplier
                )
                + [{"role": "user", "content": eval_prompt}],
            )
            for idx, eval_prompt in self.eval_set.eval_prompt(eval_size)
        ]
        eval_idxs, prompts = zip(*idxs_prompts)

        return eval_idxs, prompts

    def load_testdata(self, eval_idxs: list[int] | None) -> list[str]:
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
