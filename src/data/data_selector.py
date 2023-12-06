from .dataloader import load_data


def select_data(data_name, train=True):
    train_data, val_data, test_data = load_data(data_name)
    if not train:
        return test_data
    return val_data, train_data