import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_data(train_path, test_path, store_path):
    """
    Load the train, test, and store datasets with specified parameters.
    
    Args:
        train_path (str): Path to the training dataset.
        test_path (str): Path to the test dataset.
        store_path (str): Path to the store dataset.
        
    Returns:
        tuple: DataFrames for train, test, and store datasets.
    """
    try:
        train = pd.read_csv(train_path, parse_dates=True, low_memory=False, index_col=0)
        test = pd.read_csv(test_path, parse_dates=True, low_memory=False, index_col=0)
        store = pd.read_csv(store_path, low_memory=False)
        logger.info("Data loaded successfully")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e
    
    return train, test, store