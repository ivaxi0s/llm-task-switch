import sys
import os
from pathlib import Path
import random
from tqdm import tqdm
import json
import numpy as np

MAIN_PATH = Path(os.path.dirname(os.path.realpath(__file__)))

from src.tools.args import ModelArgs, EvalArgs
from src.tools.tools import set_seeds, get_default_device
from src.tools.saving import base_path_creator
from src.data.data_selector import select_data
from src.utils.template import template
from src.inference.predict import predict
from src.utils.eval_metric import evaluate
from src.inference.models import HFModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from src.data.dataloader import PromptLoader

load_dotenv()
import os


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

    # Load the incontext data
    # Specify shuffled
    # _, incontext_data = select_data(eval_args.incontext_data_name, datapath=MAIN_PATH)
    # # random.shuffle(incontext_data)  # This should be seeded to be deterministic
    # # incontext_examples = incontext_data[: eval_args.num_examples]

    # # Load the eval data
    # test_data = select_data(eval_args.eval_data_name, train=False)
    # if not eval_args.test_size is None and eval_args.test_size < len(test_data):
    #     print(f"Limiting test set size to: {eval_args.test_size}")
    #     test_data = random.sample(test_data, eval_args.test_size)

    pl = PromptLoader()

    # get model predictions
    model_output = base_path / "predictions.json"

    if model_output.is_file() and not core_args.force_rerun:
        print("Loading predictions from cache")
    else:
        # Run inference
        model = None
        if core_args.model_name == "mistral-7b":
            model = HFModel(
                device=get_default_device(core_args.gpu_id), model_name="mistral-7b"
            )
        # Load model into GPU

        prompts = []
        predictions = []

        for prompt in tqdm(
            pl.load_prompt(
                incontext=eval_args.incontext_data_name,
                eval=eval_args.eval_data_name,
                num_examples=eval_args.num_examples,
            )
        ):
            # Store the prompt
            prompts.append(prompt)
            # Get the prediction
            if not eval_args.no_predict:
                predictions.append(predict(core_args.model_name, prompt, model))

        # for sample in tqdm(test_data):
        #     # Get a new in context example, so that results aren't biased towards
        #     # a particular in context example
        #     incontext_examples = random.sample(incontext_data, eval_args.num_examples)
        #     prompt = template(eval_args, sample, incontext_examples)
        #     prompts.append(prompt)
        #     if not eval_args.no_predict:
        #         predictions.append(predict(core_args.model_name, prompt, model))

        # Save the prompts
        with open(base_path / "prompts.json", "w") as f:
            json.dump(prompts, f)
        # Save the predictions
        if not eval_args.no_predict:
            with open(model_output, "w") as f:
                json.dump(predictions, f)

    # Evaluate the performance
    test_data = pl.load_testdata(eval_args.eval_data_name)
    print(evaluate(model_output, test_data, eval_args.eval_data_name))
