from .dataloader import load_data


def select_data(args, train=True):
    train_data, val_data, test_data = load_data(args.data_name)
    if not train:
        return test_data
    return val_data, train_data