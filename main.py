import sys
import os
import random
from tqdm import tqdm
import json

from src.tools.args import core_args, eval_args
from src.tools.tools import set_seeds
from src.tools.saving import base_path_creator
from src.data.data_selector import load_data
from src.utils.template import template
from src.inference.predict import predict
from src.utils.eval_metric import evaluate

if __name__ == "__main__":

    # get command line arguments
    core_args, _ = core_args()
    eval_args, _ = eval_args()

    print(core_args)
    print(eval_args)

    set_seeds(core_args.seed)
    base_path = base_path_creator(core_args, eval_args)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the incontext data
    incontext_data, _, _ = load_data(eval_args.incontext_data_name)
    random.shuffle(incontext_data)
    incontext_examples = incontext_data[:eval_args.num_examples]

    # Load the eval data
    test_data = load_data(eval_args.eval_data_name, train=False)

    # get model predictions
    fname = f'{base_path}/predictions.json'
    if not os.path.isfile(fname):
        # Run inference
        predictions = []
        for sample in tqdm(test_data):
            prompt = template(eval_args, sample, incontext_examples)
            predictions.append(predict(core_args.model_name, prompt))
        
        # Save the predictions
        fname = f'{base_path}/predictions.json'
        with open(fname, 'w') as f:
            json.dump(predictions, f)
    else:
        # load predictions from cache
        with open(fname, 'r') as f:
            predictions = json.load(f)


    # Evaluate the performance
    print(evaluate(fname, test_data, eval_args.eval_data_name))
