import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_data(train_path, test_path, store_path):
    """
    Load the train, test, and store datasets.
    Args:
        train_path (str): Path to the training dataset.
        test_path (str): Path to the test dataset.
        store_path (str): Path to the store dataset.
    Returns:
        tuple: DataFrames for train, test, and store datasets.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    store = pd.read_csv(store_path)
    logger.info("Data loaded successfully")
    return train, test, store