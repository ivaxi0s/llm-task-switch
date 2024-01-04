import sys
import os
from pathlib import Path
from tqdm import tqdm
import json

MAIN_PATH = Path(os.path.dirname(os.path.realpath(__file__)))

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
    model_output = base_path / "predictions.json"
    if model_output.is_file() and not core_args.force_rerun:
        print("Loading predictions from cache")

    # ELSE: run inference using the model
    else:
        # Initialise / load model
        model = get_model(core_args.model_name, core_args.gpu_id)

        predictions = []

        print("Loading prompts")
        if eval_args.iterative:
            prompts = pl.load_prompt_iterative(num_examples=eval_args.num_examples)
        else:
            prompts = pl.load_prompt(num_examples=eval_args.num_examples)

        for i in tqdm(range(0, len(prompts), core_args.batchsize)):
            prompt_batch = prompts[i : i + core_args.batchsize]
            # Get the prediction
            if not eval_args.no_predict:
                if eval_args.iterative:
                    predictions.extend(model.predict_batch_iteratively(prompt_batch))
                else:
                    predictions.extend(model.predict_batch(prompt_batch))

        # Save the prompts
        with open(base_path / "prompts.json", "w") as f:
            json.dump(prompts, f)
        # Save the predictions
        if not eval_args.no_predict:
            with open(model_output, "w") as f:
                json.dump(predictions, f)

    # Evaluate the performance
    test_data = pl.load_testdata()
    print(evaluate(model_output, test_data, eval_args.eval_data_name))
