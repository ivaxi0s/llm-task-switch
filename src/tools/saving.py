import os

def base_path_creator(core_args, eval_args, create=True):
    path = '.'
    path = next_dir(path, 'experiments', create=create)
    path = next_dir(path, core_args.model_name, create=create)
    path = next_dir(path, f'eval_data_{eval_args.eval_data_name}', create=create)
    path = next_dir(path, f'incontext_data_{eval_args.incontext_data_name}', create=create)
    path = next_dir(path, f'num_examples_{eval_args.num_examples}', create=create)
    return path

def next_dir(path, dir_name, create=True):
    if not os.path.isdir(f'{path}/{dir_name}'):
        if create:
            os.mkdir(f'{path}/{dir_name}')
        else:
            raise ValueError ("provided args do not give a valid model path")
    path += f'/{dir_name}'
    return path