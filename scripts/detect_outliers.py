# data_quality.py

import pandas as pd

def detect_outliers(df):
    """
    Detect outliers in the dataframe.
    Args:
        df (pd.DataFrame): The dataframe to check for outliers.
    Returns:
        dict: Dictionary containing outlier information for each numerical feature.
    """
    outliers = {}
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    
    for feature in numerical_features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_indices = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)].index
        outliers[feature] = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'num_outliers': len(outlier_indices),
            'outlier_indices': outlier_indices
        }
    
    return outliers