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

    set_seeds(core_args.seed)
    base_path = base_path_creator(MAIN_PATH, core_args, eval_args)

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

    # IF model predictions exist, load them
    model_output_file = base_path / "predictions.json"
    if not core_args.force_rerun and model_output_file.is_file():
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

        print(prompts[0])

        # Save the prompts
        with open(base_path / "prompts.json", "w") as f:
            json.dump(prompts, f)
        # Save the idxs
        with open(base_path / EVAL_IDXS_FILE, "w") as f:
            json.dump(eval_idxs, f)

        if not eval_args.no_predict or eval_args.likelihoods:
            # Initialise / load model
            print(f"Loading model: {core_args.model_name}")
            model = get_model(core_args.model_name, core_args.gpu_id)

        if not eval_args.no_predict:
            # Get predictions on test set
            predictions = []
            for i in tqdm(range(0, len(prompts), core_args.batchsize)):
                # batch prompts
                prompt_batch = prompts[i : i + core_args.batchsize]
                breakpoint()
                if eval_args.iterative:
                    predictions.extend(model.predict_batch_iteratively(prompt_batch))
                else:
                    predictions.extend(model.predict_batch(prompt_batch))

            # Save the predictions
            if not eval_args.no_predict:
                with open(model_output_file, "w") as f:
                    json.dump(predictions, f)
        
        if eval_args.likelihoods:
            print("Computing likelihoods")
            if not eval_args.iterative:
                raise ValueError("Likelihoods can only be computed for iterative prompts")
            # Load the predictions for 0 in-context examples
            baseline_predictions_file = base_path.parent.parent / "num_examples_0" / "iterative"/ "predictions.json"
            with open(baseline_predictions_file, "r") as f:
                baseline_predictions: list[str] = json.load(f)
            if len(baseline_predictions) != len(prompts):
                raise ValueError("Baseline predictions and prompts are not the same length")

            baseline_likelihoods = []
            for response, prompt in tqdm(zip(baseline_predictions, prompts), total=len(prompts)):
                baseline_likelihoods.append(model.response_probabilities(prompt, response))
            
            # Save the baseline probabilities using pickle
            pickle.dump(baseline_likelihoods, open(base_path / "baseline_probabilities.pkl", "wb"))
            # np.save(base_path / "baseline_probabilities.npy", baseline_likelihoods)

    # Evaluate the performance
    # Check if eval idxs exists (if not, use the entire test set)
    if (base_path / EVAL_IDXS_FILE).is_file():
        with open(base_path / EVAL_IDXS_FILE, "r") as f:
            eval_idxs = json.load(f)
            print(f"Loaded eval idxs: {len(eval_idxs)}")
    else:
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
