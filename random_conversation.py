"""Evaluate task-switch interference with a "random" conversation history

This is another way of identifying the baseline robustness of a model to task-switch interference.

Method:

- load model
- load prompts: [history, target]
    - history: [user0, user1, ...]
    - target: [user_T]
    - discard history, but keep the target (we will "randomly" generate the history)

for prompt in prompts:
    *_, target = prompt
    conversation = []

    for _ in range(T):
        # 1. History Task:
        # a. Generate a random user prompt
        user = model.generate_user_prompt(conversation)

        # b. Generate the system response
        conversation = conversation + user
        model.predice(conversation)

        conversation = conversation + target

        # 2. Target Task: Generate the system response for the task
        model.predict(conversation + target)
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
import random

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
if core_args.batchsize != 1:
    raise ValueError("This script is only for batchsize 1")
if eval_args.eval_data_name != eval_args.incontext_data_name:
    raise ValueError("For simplicity in data analysis later, set eval=incontext")

set_seeds(core_args.seed)
base_path = base_path_creator(MAIN_PATH, core_args, eval_args)
base_path = base_path / "random_conversation"
base_path.mkdir(parents=True, exist_ok=True)
print("--> Output path: ", base_path)


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
    # Remove the history and keep the target
    prompts = [[prompt[-1]] for prompt in prompts]

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
        # Load model
        print(f"Loading model: {core_args.model_name}")
        model = get_model(core_args.model_name, core_args.gpu_id)

        random_history = model.generate_random_history(seed=core_args.seed)

        # Create a conversation history for each prompt
        # by randomly selecting eval_args.num_examples from random_history
        prompts = [
            [
                utterance
                for user_assistant in random.sample(
                    random_history, eval_args.num_examples
                )
                for utterance in user_assistant
            ]
            + prompt
            for prompt in prompts
        ]

        # Print the first prompt
        print("First prompt\n")
        pprint.pprint(prompts[0])

        # Save the prompts with history
        with open(base_path / "prompts_with_history.json", "w") as f:
            json.dump(prompts, f)

        # Get predictions on test set
        predictions = []

        for prompt in tqdm(prompts):
            *conversation_history, target = prompt
            target_responses: list[str] = []  # response after each turn in the history

            # Get the response for the target after each turn in the history
            for idx in range(0, len(conversation_history) + 1, 2):
                conversation = [[*conversation_history[:idx], target]]
                target_responses.extend(model.predict_batch_iteratively(conversation))

            predictions.append(target_responses)
            # breakpoint()

        # Save the predictions
        predictions = list(zip(*predictions))  # transpose the list
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
