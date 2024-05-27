import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

class DataPreprocessing:
    def __init__(self, train_path, test_path, store_path):
        self.train_path = train_path
        self.test_path = test_path
        self.store_path = store_path
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        self.train = pd.read_csv(self.train_path)
        self.test = pd.read_csv(self.test_path)
        self.store = pd.read_csv(self.store_path)
        self.logger.info("Data loaded successfully")

    def merge_data(self):
        self.train = pd.merge(self.train, self.store, on='Store')
        self.test = pd.merge(self.test, self.store, on='Store')
        self.logger.info("Data merged successfully")

    def preprocess(self):
        # Handle missing values
        self.train.fillna(0, inplace=True)
        self.test.fillna(0, inplace=True)
        # Extract datetime features
        self.train['Date'] = pd.to_datetime(self.train['Date'])
        self.test['Date'] = pd.to_datetime(self.test['Date'])
        self.train['Year'] = self.train['Date'].dt.year
        self.train['Month'] = self.train['Date'].dt.month
        self.train['Day'] = self.train['Date'].dt.day
        self.train['WeekOfYear'] = self.train['Date'].dt.isocalendar().week
        self.test['Year'] = self.test['Date'].dt.year
        self.test['Month'] = self.test['Date'].dt.month
        self.test['Day'] = self.test['Date'].dt.day
        self.test['WeekOfYear'] = self.test['Date'].dt.isocalendar().week
        self.logger.info("Preprocessing done")

    def scale_data(self):
        scaler = StandardScaler()
        self.train[['Sales', 'Customers']] = scaler.fit_transform(self.train[['Sales', 'Customers']])
        self.test[['Sales', 'Customers']] = scaler.transform(self.test[['Sales', 'Customers']])
        self.logger.info("Data scaling done")

    def save_processed_data(self):
        self.train.to_csv('data/processed/train_processed.csv', index=False)
        self.test.to_csv('data/processed/test_processed.csv', index=False)
        self.logger.info("Processed data saved")

    def run(self):
        self.load_data()
        self.merge_data()
        self.preprocess()
        self.scale_data()
        self.save_processed_data()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    processor = DataPreprocessing('data/raw/train.csv', 'data/raw/test.csv', 'data/raw/store.csv')
    processor.run()