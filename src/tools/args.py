import argparse

def core_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument('--model_name', type=str, default='gpt3.5', help='LLM to evaluate')
    commandLineParser.add_argument('--openai_key', type=str, default='', help='openai key to access models')
    commandLineParser.add_argument('--gpu_id', type=int, default=0, help='select specific gpu')
    commandLineParser.add_argument('--seed', type=int, default=1, help='select seed')
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    return commandLineParser.parse_known_args()


def eval_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument('--incontext_data_name', type=str, default='rt', help='dataset for incontext examples')
    commandLineParser.add_argument('--eval_data_name', type=str, default='rt', help='dataset to evaluate performance on')
    commandLineParser.add_argument('--num_examples', type=int, default=0, help='Number of in context examples to provide')
    return commandLineParser.parse_known_args()