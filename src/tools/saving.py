import os
from pathlib import Path
from src.tools.args import ModelArgs, EvalArgs


def base_path_creator(
    main_path: Path, core_args: ModelArgs, eval_args: EvalArgs, create=True
) -> Path:
    """Create directory structure for saving model outputs

    Model outputs are saved in the following structure:
    experiments / model_name / eval dataset / incontext_dataset / num_examples
    """

    output_path = (
        main_path
        / "experiments"
        / core_args.model_name
        / f"eval_data_{eval_args.eval_data_name}"
        / f"incontext_data_{eval_args.incontext_data_name}"
        / f"num_examples_{eval_args.num_examples}"
    )

    if eval_args.iterative:
        output_path = output_path / "iterative"

    output_path.mkdir(parents=create, exist_ok=True)
    print("Saving to output path: ", output_path)

    return output_path
