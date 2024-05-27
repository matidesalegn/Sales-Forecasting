# EDA Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class EDA:
    def __init__(self, data_path):
        self.data_path = data_path
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        self.logger.info("Data loaded for EDA")

    def plot_sales_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data['Sales'], kde=True)
        plt.title('Sales Distribution')
        plt.xlabel('Sales')
        plt.ylabel('Frequency')
        plt.show()

    def plot_sales_over_time(self):
        plt.figure(figsize=(12, 8))
        self.data.groupby('Date')['Sales'].sum().plot()
        plt.title('Total Sales Over Time')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.show()

    def plot_correlation_heatmap(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm')
        plt.title('Feature Correlation Heatmap')
        plt.show()

    def run(self):
        self.load_data()
        self.plot_sales_distribution()
        self.plot_sales_over_time()
        self.plot_correlation_heatmap()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eda = EDA('data/processed/train_processed.csv')
    eda.run()