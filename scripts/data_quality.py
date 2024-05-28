# data_quality.py

import logging
import pandas as pd

logger = logging.getLogger(__name__)

def check_data_quality(df):
    """
    Check the quality of the dataset by inspecting the first few rows,
    data types, and missing values.
    Args:
        df (pd.DataFrame): The dataset to check.
    Returns:
        dict: A dictionary containing data quality information.
    """
    quality_info = {
        "head": df.head(),
        "dtypes": df.dtypes,
        "missing_values": df.isnull().sum()
    }
    logger.info("Data quality check completed")
    return quality_info