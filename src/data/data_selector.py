from .dataloader import load_data
from pathlib import Path

DATA_PATH = Path("data")


# This is bogus function
def select_data(data_name, train=True, datapath: Path = DATA_PATH):
    # TODO: add arg to limit the test set size

    # Check if the data exists in the MAIN_PATH

    train_data, val_data, test_data = load_data(data_name)

    # Save

    if not train:
        return test_data
    return val_data, train_data
