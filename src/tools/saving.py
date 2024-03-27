import os
from pathlib import Path
from src.tools.args import ModelArgs, EvalArgs


def base_path_creator(
    main_path: Path,
    core_args: ModelArgs,
    eval_args: EvalArgs,
    create=True,
    converse=False,
) -> Path:
    """Create directory structure for saving model outputs

    Model outputs are saved in the following structure:
    experiments / model_name / eval dataset / incontext_dataset / num_examples

    Optionally, `iterative` is added to the path if iterative inference is used.
    Optionally, if `converse` is True, it is added to the path.
    Optionally, if `seed` is not 1, then create a new folder `/seed_{seed}`
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

    if converse:
        output_path = output_path / "converse"

    if core_args.seed != 1:
        output_path = output_path / f"seed_{core_args.seed}"

    output_path.mkdir(parents=create, exist_ok=True)
    print("Saving to output path: ", output_path)

    return output_path
