import argparse
from dataclasses import dataclass


@dataclass
class ModelArgs:
    """Class for storing model details"""

    model_name: str
    gpu_id: int
    seed: int
    force_cpu: bool
    force_rerun: bool

    @staticmethod
    def argparse() -> "ModelArgs":
        """Parse arguments and return an instance of ModelArgs"""
        commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
        commandLineParser.add_argument(
            "--model_name", type=str, default="mistral-7b", help="LLM to evaluate"
        )  # options: "gpt3.5, mistral-7b"
        commandLineParser.add_argument(
            "--gpu_id", type=int, default=0, help="select specific gpu"
        )
        commandLineParser.add_argument(
            "--seed", type=int, default=1, help="select seed"
        )
        commandLineParser.add_argument(
            "--force_cpu", action="store_true", help="force cpu use"
        )
        commandLineParser.add_argument(
            "--force_rerun", action="store_true", help="force rerun"
        )

        parsedArgs, _ = commandLineParser.parse_known_args()

        return ModelArgs(
            model_name=parsedArgs.model_name,
            gpu_id=parsedArgs.gpu_id,
            seed=parsedArgs.seed,
            force_cpu=parsedArgs.force_cpu,
            force_rerun=parsedArgs.force_rerun,
        )


@dataclass
class EvalArgs:
    """Class for storing evaluation details"""

    incontext_data_name: str
    eval_data_name: str  # Evaluation dataset
    num_examples: int
    no_predict: bool
    test_size: int | None  # Number of examples to evaluate on (if > test set size, use test set size)

    @staticmethod
    def argparse() -> "EvalArgs":
        """Parse arguments and return an instance of EvalArgs"""
        commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
        commandLineParser.add_argument(
            "--incontext_data_name",
            type=str,
            default="gigaword",
            help="dataset for incontext examples",
        )
        commandLineParser.add_argument(
            "--eval_data_name",
            type=str,
            default="gigaword",
            help="dataset to evaluate performance on",
        )
        commandLineParser.add_argument(
            "--num_examples",
            type=int,
            default=0,
            help="Number of in context examples to provide",
        )
        commandLineParser.add_argument(
            "--no_predict",
            action="store_true",
            help="Do not predict (and do not save predictions)",
        )
        commandLineParser.add_argument(
            "--test_size",
            type=int,
            default=None,
            help=(
                "Number of examples to evaluate on (if > test set size, use test set size)"
                "If None, use the entire test set"
            ),
        )

        parsedArgs, _ = commandLineParser.parse_known_args()
        return EvalArgs(
            incontext_data_name=parsedArgs.incontext_data_name,
            eval_data_name=parsedArgs.eval_data_name,
            num_examples=parsedArgs.num_examples,
            no_predict=parsedArgs.no_predict,
            test_size=parsedArgs.test_size,
        )
