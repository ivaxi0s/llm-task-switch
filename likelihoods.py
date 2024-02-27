"""There are three different likelihoods that can be computed:

Required Flags:
    --iterative
    --likelihoods

Args:
    --num_examples <int>
    --eval_size (generally 100)
    --eval_data_name <dataset>
    --incontext_data_name <dataset>
    --model_name <model>

When running the script, we calculate the likelihoods for the following three cases:
1. P(r_0 | u, h_L) : The likelihood of the original response given the prompt and the final context
2. P(r_L | u, h_L) : The likelihood of the final prediction given the prompt and the final context
3. P(r_ref | u, h_L) : The likelihood of the reference response given the prompt and the final context

The above probabilities are saved in a pickle file and can be used to calculate the following:
1. `base_probabilties.pkl`
2. `final_probabilities.pkl`
3. `ref_probabilities.pkl`


NOTE that when running zero-shot (i.e. with 0 in-context examples), 
the likelihoods are hence calculated for the following:
1. P(r_0 | u) : The likelihood of the original response given the prompt
2. P(r_0 | u) : (The same as above, because r_L = r_0 when h_L is empty)
3. P(r_ref | u) : The likelihood of the reference response given the prompt

These can be combined to calculate the following:

_zero-shot sensitivity_: Sensitivity in predicting the original response
P(r_0 | u, h_L)
----------------
P(r_0 | u)

_confidence sensitivity_: Confidence in the final prediction
P(r_L | u, h_L)
----------------
P(r_0 | u)

_loss sensitivity_: Predicting the reference response
P(r_ref | u, h_L)
----------------
P(r_ref | u)
"""

import sys
import os
from pathlib import Path
from tqdm import tqdm  # type: ignore
import json
import numpy as np
import pickle

from src.tools.args import ModelArgs, EvalArgs
from src.tools.tools import set_seeds
from src.tools.saving import base_path_creator
from src.eval.eval_metric import evaluate
from src.inference.models import get_model, HFModel
from dotenv import load_dotenv
from src.data.promptloader import PromptLoader

MAIN_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
EVAL_IDXS_FILE = "eval_idxs.json"

load_dotenv()
# print(os.environ["HF_HOME"])

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

base_ll_filepath = base_path / "base_probabilities.pkl"
final_ll_filepath = base_path / "final_probabilities.pkl"
ref_ll_filepath = base_path / "ref_probabilities.pkl"


def get_predictions_from_idxs(idxs: list[int], num_examples: int):
    results_path = (
        base_path.parent.parent / f"num_examples_{num_examples}" / "iterative"
    )
    idx_file = results_path / EVAL_IDXS_FILE
    predictions_file = results_path / "predictions.json"

    with open(idx_file, "r") as f:
        pred_idxs = json.load(f)
    with open(predictions_file, "r") as f:
        predictions = json.load(f)

    # Find the relative idx in pred_idxs
    pred_relative_idxs = [pred_idxs.index(i) for i in idxs]

    # Get the predictions corresponding to the idxs
    return [predictions[i] for i in pred_relative_idxs]


if (
    not core_args.force_rerun
    and base_ll_filepath.is_file()
    and final_ll_filepath.is_file()
    and ref_ll_filepath.is_file()
):
    print("Loading predictions from cache")

# ELSE: run inference using the model
else:
    print("Loading prompts and context history")
    eval_idxs, prompts = pl.load_prompt_iterative(
        num_examples=eval_args.num_examples, eval_size=eval_args.eval_size
    )

    # Warn if the number of examples is very large
    if eval_args.num_examples > 100:
        print(f"WARNING: {eval_args.num_examples}  may take a long time to compute.")

    # Save the idxs used to calculate the log-likelihoods
    with open(base_path / ("ll-" + EVAL_IDXS_FILE), "w") as f:
        json.dump(eval_idxs, f)

    # 1. Load the predictions for 0 in-context examples
    baseline_predictions = get_predictions_from_idxs(eval_idxs, 0)

    # 2. Load the predictions for the current number of examples
    final_predictions = get_predictions_from_idxs(eval_idxs, eval_args.num_examples)

    # 3. Load the reference responses
    reference_data = pl.load_likelihood_reference(eval_idxs)

    # Assert that the lengths are the same
    assert (
        len(baseline_predictions)
        == len(final_predictions)
        == len(reference_data)
        == len(prompts)
    )

    # Print an example prompt and set of responses
    print("Example:")
    print("prompt:\n", prompts[0])
    print("base response:\n", baseline_predictions[0])
    print("final response:\n", final_predictions[0])
    print("ref response:\n", reference_data[0])

    # Initialise / load model
    print(f"Loading model: {core_args.model_name}")
    model = get_model(core_args.model_name, core_args.gpu_id)
    if not isinstance(model, HFModel):
        raise ValueError("Model must be an HFModel")

    print("Computing likelihoods")
    baseline_likelihoods = []
    final_likelihoods = []
    ref_likelihoods = []
    for ref_data, final_pred, base_pred, prompt in tqdm(
        zip(reference_data, final_predictions, baseline_predictions, prompts),
        total=len(prompts),
    ):
        # Compute the likelihoods for each
        baseline_likelihoods.append(model.response_probabilities(prompt, base_pred))
        final_likelihoods.append(model.response_probabilities(prompt, final_pred))
        ref_likelihoods.append(model.response_probabilities(prompt, ref_data))

    # Save the baseline probabilities using pickle
    pickle.dump(baseline_likelihoods, open(base_ll_filepath, "wb"))
    pickle.dump(final_likelihoods, open(final_ll_filepath, "wb"))
    pickle.dump(ref_likelihoods, open(ref_ll_filepath, "wb"))
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
