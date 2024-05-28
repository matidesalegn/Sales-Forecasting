import pandas as pd

def check_data_types(df):
    """
    Check the data types of features in the dataframe.
    Args:
        df (pd.DataFrame): The dataframe to check.
    Returns:
        dict: Dictionary containing data types for each feature.
    """
    data_types = df.dtypes.to_dict()
    return data_types