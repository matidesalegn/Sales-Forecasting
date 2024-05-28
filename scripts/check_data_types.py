import logging
import pandas as pd

logger = logging.getLogger(__name__)

def check_data_types(df):
    """
    Check the data types of features in the dataframe.
    Args:
        df (pd.DataFrame): The dataframe to check.
    Returns:
        dict: Dictionary containing data types for each feature.
    """
    data_types = df.dtypes.to_dict()
    logger.info("Data types checked")
    return data_types