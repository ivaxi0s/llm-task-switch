import sys
import os
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
import pickle

MAIN_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
EVAL_IDXS_FILE = "eval_idxs.json"


from src.tools.args import ModelArgs, EvalArgs
from src.tools.tools import set_seeds
from src.tools.saving import base_path_creator
from src.utils.eval_metric import evaluate
from src.inference.models import get_model
from dotenv import load_dotenv
from src.data.dataloader import PromptLoader

load_dotenv()
# print(os.environ["HF_HOME"])


if __name__ == "__main__":
    # get command line arguments
    core_args = ModelArgs.argparse()
    eval_args = EvalArgs.argparse()

    print(core_args)
    print(eval_args)

    if not eval_args.likelihoods:
        raise ValueError("This script is only for use with --likelihoods flag")
    if not eval_args.iterative:
        raise ValueError("Likelihoods can only be computed for iterative prompts")
    set_seeds(core_args.seed)
    base_path = base_path_creator(MAIN_PATH, core_args, eval_args)
    model_output_file = base_path / "predictions.json"

    # Append the command run to a file
    attack_cmd_file = MAIN_PATH / "CMDs" / "attack.cmd"
    attack_cmd_file.parent.mkdir(parents=True, exist_ok=True)
    with open(attack_cmd_file, "a") as f:
        f.write(" ".join(sys.argv) + "\n")

    # Load dataset
    print("Loading prompt loader")
    pl = PromptLoader(
        eval=eval_args.eval_data_name, incontext=eval_args.incontext_data_name
    )

    ll_filepath = base_path / "base_probabilities.pkl"

    if not core_args.force_rerun and ll_filepath.is_file():
        print("Loading predictions from cache")

    # ELSE: run inference using the model   
    else:
        print("Loading prompts")
        if eval_args.iterative:
            eval_idxs, prompts = pl.load_prompt_iterative(
                num_examples=eval_args.num_examples, eval_size=eval_args.eval_size
            )
        else:
            eval_idxs, prompts = pl.load_prompt(
                num_examples=eval_args.num_examples, eval_size=eval_args.eval_size
            )

        # Save the idxs used to calculate the log-likelihoods
        with open(base_path / ("ll-" + EVAL_IDXS_FILE), "w") as f:
            json.dump(eval_idxs, f)

        # Initialise / load model
        print(f"Loading model: {core_args.model_name}")
        model = get_model(core_args.model_name, core_args.gpu_id)

        print("Computing likelihoods")

        # Load the predictions for 0 in-context examples
        baseline_predictions_file = (
            base_path.parent.parent
            / "num_examples_0"
            / "iterative"
            / "predictions.json"
        )
        with open(baseline_predictions_file, "r") as f:
            baseline_predictions: list[str] = json.load(f)

        # Load the indexes for the 0 in-context examples
        baseline_idxs_file = (
            base_path.parent.parent / "num_examples_0" / "iterative" / EVAL_IDXS_FILE
        )
        with open(baseline_idxs_file, "r") as f:
            baseline_idxs = json.load(f)

        # Get the baseline predictions corresponding to the current eval_idxs
        baseline_relative_idxs = [baseline_idxs.index(i) for i in eval_idxs]
        baseline_predictions = [baseline_predictions[i] for i in baseline_relative_idxs]

        if len(baseline_predictions) != len(prompts):
            raise ValueError("Baseline predictions and prompts are not the same length")

        # Print an example prompt and response
        print("Example prompt and response")
        print(prompts[0])
        print(baseline_predictions[0])

        baseline_likelihoods = []
        for response, prompt in tqdm(
            zip(baseline_predictions, prompts), total=len(prompts)
        ):
            baseline_likelihoods.append(model.response_probabilities(prompt, response))

        # Save the baseline probabilities using pickle
        pickle.dump(
            baseline_likelihoods,
            open(ll_filepath, "wb"),
        )
        # np.save(base_path / "baseline_probabilities.npy", baseline_likelihoods)

    # Evaluate the performance
    # Check if eval idxs exists (if not, use the entire test set)
    if (base_path / EVAL_IDXS_FILE).is_file():
        with open(base_path / EVAL_IDXS_FILE, "r") as f:
            eval_idxs = json.load(f)
            print(f"Loaded eval idxs: {len(eval_idxs)}")
    else:
        print("No eval idxs file found. Defaulting to entire test set.")
        eval_idxs = None

    reference_data = pl.load_testdata(eval_idxs)
    print(
        evaluate(
            model_output_file,
            reference_data,
            eval_args.eval_data_name,
            # Revaluate if predictions are re-run
            # use_cached=core_args.force_rerun,
            use_cached=False,
        )
    )
