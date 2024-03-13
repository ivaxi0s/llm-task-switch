"""Evaluate task-switch inference *without* teacher-forcing response

In our original paper,...
The script runs the iterative user-prompt tasks but without teacher-forcing the result.


Method:

- Load the model
- Load the prompts: [history, target]
    - history: [user0, user1, ...]
    - target: [user_T]
- Assume: there are no system responses in the turns
This is run like so:

First pass is used to generate the system responses

for prompt in prompts:
    *history, target = prompt

    # Add an empty string to the start of the history
    # This is the zero-shot history
    history = ["", *history]
    conversation = []
    
    for h in history:
        conversation = h + conversation
        # 1. Generate the system response for the history
        model.predict(conversation)

        # 2. Generate the system response for the task
        conversation = conversation + target
    
        model.predict(conversation)
"""

import sys
import os
from pathlib import Path
from tqdm import tqdm  # type: ignore
import json
import pprint

from src.tools.args import ModelArgs, EvalArgs
from src.tools.tools import set_seeds
from src.tools.saving import base_path_creator
from src.eval.eval_metric import evaluate, evaluate_converse
from src.inference.models import get_model
from dotenv import load_dotenv
from src.data.promptloader import PromptLoader

MAIN_PATH = Path(os.path.dirname(os.path.realpath(__file__)))

# Append the command run to a file
attack_cmd_file = MAIN_PATH / "CMDs" / "attack.cmd"
attack_cmd_file.parent.mkdir(parents=True, exist_ok=True)
with open(attack_cmd_file, "a") as f:
    f.write(" ".join(sys.argv) + "\n")

EVAL_IDXS_FILE = "eval_idxs.json"


load_dotenv()
# print(os.environ["HF_HOME"])

# get command line arguments
core_args = ModelArgs.argparse()
eval_args = EvalArgs.argparse()

print(core_args)
print(eval_args)

if not eval_args.iterative:
    raise ValueError("This script is only for iterative inference")
if not core_args.batchsize == 1:
    raise ValueError("This script is only for batchsize 1")

set_seeds(core_args.seed)
base_path = base_path_creator(MAIN_PATH, core_args, eval_args, converse=True)
print("Output path: ", base_path)

# Load dataset
print("Loading prompt loader")
pl = PromptLoader(
    eval=eval_args.eval_data_name, incontext=eval_args.incontext_data_name
)

# IF model predictions exist, load them
model_output_file = base_path / "predictions.json"
if not core_args.force_rerun and model_output_file.is_file():
    print("Loading predictions from cache")
    # predictions = json.load(model_output_file)

# ELSE: run inference using the model
else:
    print("Loading prompts")
    eval_idxs, prompts = pl.load_prompt_iterative(
        num_examples=eval_args.num_examples, eval_size=eval_args.eval_size
    )

    # Keep the role="user" from the prompts
    # i.e. remove the system responses
    prompts = [
        [turn for turn in prompt if turn["role"] == "user"] for prompt in prompts
    ]

    # Print the first prompt
    print("First prompt\n")
    pprint.pprint(prompts[0])

    # Save the prompts
    with open(base_path / "prompts.json", "w") as f:
        json.dump(prompts, f)
    # Save the idxs
    with open(base_path / EVAL_IDXS_FILE, "w") as f:
        json.dump(eval_idxs, f)

    if not eval_args.no_predict:
        # Initialise / load model
        print(f"Loading model: {core_args.model_name}")
        model = get_model(core_args.model_name, core_args.gpu_id)

        # Get predictions on test set
        predictions = []

        for prompt in tqdm(prompts):
            *history, target = prompt

            conversation_history: list[dict] = []
            target_responses: list[str] = []  # response after each turn in the history

            # zero shot response to target:
            # target_responses.extend(model.predict_batch_iteratively([target]))

            # Get the response for each turn in the history
            for h in history:
                conversation_history.append(h)
                (response,) = model.predict_batch_iteratively([conversation_history])
                conversation_history.append({"role": "assistant", "content": response})

            # Get the response for the target after each turn in the history
            for idx in range(0, len(conversation_history) + 1, 2):
                target_responses.extend(
                    model.predict_batch_iteratively(
                        [[*conversation_history[:idx], target]]
                    )
                )

            predictions.append(target_responses)
            # breakpoint()

        # breakpoint()
        # Save the predictions
        predictions = list(zip(*predictions))  # type: ignore
        assert len(predictions) == eval_args.num_examples + 1

        with open(model_output_file, "w") as f:
            json.dump(predictions, f)

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

results = evaluate_converse(
    model_output_file,
    reference_data,
    use_cached=False,
)

print(results)
