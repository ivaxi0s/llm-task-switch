"""Script to santise prompts

There was a bug where the correct test set idxs for inference
were not being stored. This resulted in the wrong set of idxs
being used for evaluation. 

NOTE: the idxs are stored in a file called eval_idxs.json
NOTE: we assert that we are running with the --iterative flag

This script will load the idxs from eval_idxs.json.
If the number of idxs is the same as eval_args.eval_size,
then we will re-run the prompt loader to generate the _correct_
set of idxs.
"""

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

core_args = ModelArgs.argparse()
eval_args = EvalArgs.argparse()

print(core_args)
print(eval_args)

if not eval_args.iterative:
    raise ValueError("This script is only for use with --iterative flag")
if eval_args.eval_size is None:
    raise ValueError("This script is only for use with --eval_size")

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

print("Verifying prompt idxs")

# Load the stored eval_idxs
stored_eval_idxs = json.load(open(base_path / EVAL_IDXS_FILE, "r"))
# Get the number of idxs
num_stored_eval_idxs = len(stored_eval_idxs)
print(f"Number of stored eval idxs: {num_stored_eval_idxs}")

# If the number of stored idxs is the same as eval_args.eval_size
# then we re-run the prompt loader to get the correct idxs

if num_stored_eval_idxs == eval_args.eval_size:
    print("Re-running prompt loader to get the correct idxs")
    eval_idxs, prompts = pl.load_prompt_iterative(
        num_examples=eval_args.num_examples, eval_size=eval_args.eval_size
    )
    if not all([i == j for i, j in zip(eval_idxs, stored_eval_idxs)]):
        print(f"--- Sanitised idxs for {core_args.model_name} {eval_args} ---")
        print("Saving new idxs")
        with open(base_path / EVAL_IDXS_FILE, "w") as f:
            json.dump(eval_idxs, f)
    else:
        print("Using stored idxs")
else:
    print("Using stored idxs")
