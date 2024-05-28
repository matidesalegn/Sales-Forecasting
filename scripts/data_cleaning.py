import pandas as pd
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # Set logging level to INFO

def preprocess_data(df, store_df):
    # Merge with store data
    df = pd.merge(df, store_df, how='left', on='Store')
    
    # Handle missing values for 'Open' column
    if 'Open' in df.columns:
        logger.info("Number of missing values in 'Open' column before filling: %d", df['Open'].isnull().sum())
        df['Open'].fillna(1, inplace=True)  # Assume open if not specified
        logger.info("Number of missing values in 'Open' column after filling: %d", df['Open'].isnull().sum())
    
    # Fill missing values
    df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
    df['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
    df['CompetitionOpenSinceYear'].fillna(0, inplace=True)
    df['Promo2SinceWeek'].fillna(0, inplace=True)
    df['Promo2SinceYear'].fillna(0, inplace=True)
    df['PromoInterval'].fillna('', inplace=True)
    
    # Convert date columns to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract date features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    # Encode categorical variables
    df['StateHoliday'] = df['StateHoliday'].astype(str)
    df['StoreType'] = df['StoreType'].astype(str)
    df['Assortment'] = df['Assortment'].astype(str)
    df['PromoInterval'] = df['PromoInterval'].astype(str)
    
    logger.info("Data preprocessed successfully")
    return df
    """
    Preprocess the dataset by merging with store data, filling missing values,
    and extracting date features.
    Args:
        df (pd.DataFrame): The dataset to preprocess.
        store_df (pd.DataFrame): The store dataset.
    Returns:
        pd.DataFrame: The preprocessed dataset.
    The process of merging the store data with the training and test data is an important step in the preprocessing pipeline. The store data contains additional information about each store that can be crucial for understanding and modeling customer behavior. For instance, it includes details like StoreType, Assortment, CompetitionDistance, and other variables that provide context about the store's characteristics and competitive environment.

Here is a step-by-step explanation of why and how we preprocess the data, including merging with store data:

Merging with Store Data:

Why: The store data contains additional features about each store that are not present in the training or test data. By merging this information, we can enrich our dataset with more contextual information that might be valuable for analysis and modeling.
How: Use pd.merge to combine the datasets on the Store column. This ensures that each row in the training and test datasets has corresponding information from the store dataset.
Handling Missing Values:

Certainly! This section of code is focused on handling missing values in specific columns of the DataFrame df. Let's break down each line:

CompetitionDistance:

df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True): This line fills missing values in the 'CompetitionDistance' column with the median value of the existing data in that column. Using the median helps maintain the central tendency of the data and is less sensitive to outliers. The inplace=True parameter ensures that the changes are made directly to the DataFrame df without the need to assign it back to a variable.
CompetitionOpenSinceMonth and CompetitionOpenSinceYear:

df['CompetitionOpenSinceMonth'].fillna(0, inplace=True): This line fills missing values in the 'CompetitionOpenSinceMonth' column with 0. It's common to use 0 as a placeholder for missing categorical or numerical data when there's no other appropriate value to fill.
df['CompetitionOpenSinceYear'].fillna(0, inplace=True): Similar to the previous line, this fills missing values in the 'CompetitionOpenSinceYear' column with 0.
Promo2SinceWeek and Promo2SinceYear:

df['Promo2SinceWeek'].fillna(0, inplace=True): This line fills missing values in the 'Promo2SinceWeek' column with 0.
df['Promo2SinceYear'].fillna(0, inplace=True): Similar to the previous line, this fills missing values in the 'Promo2SinceYear' column with 0.
PromoInterval:

df['PromoInterval'].fillna('', inplace=True): This line fills missing values in the 'PromoInterval' column with an empty string ''. This might indicate that there is no specific interval for promotional activities for those missing values.
Overall, these lines ensure that the DataFrame df is properly prepared for further analysis and modeling by filling missing values in specific columns with appropriate placeholders or central tendency values. This helps prevent potential issues during analysis and ensures consistency in the dataset.

Extracting Date Features:

Convert the Date column to datetime format.
Extract useful date features such as Year, Month, Day, WeekOfYear, and DayOfWeek for further analysis and modeling.

Clean the data using preprocess_data(). Specifically, handle the 'Open' column's missing values by filling them with 1 (assuming the store is open if not specified).
Open: Fill missing values in the 'Open' column with 1, assuming that the store is open if not specified.
CompetitionDistance: Fill missing values with the median value of the column to handle outliers and ensure a reasonable central value is used.
Other Columns: Fill missing values in other columns like CompetitionOpenSinceMonth, CompetitionOpenSinceYear, Promo2SinceWeek, Promo2SinceYear, and PromoInterval with default values (0 or empty string).
Extracting Date Features:

Convert the Date column to datetime format.
Extract useful date features such as Year, Month, Day, WeekOfYear, and DayOfWeek for further analysis and modeling.
Encoding Categorical Variables:

Convert categorical columns (StateHoliday, StoreType, Assortment, and PromoInterval) to string type for consistent handling and future encoding if needed.
    """