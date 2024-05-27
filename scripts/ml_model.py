import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib
import logging

class MLModel:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        self.train = pd.read_csv(self.train_path)
        self.test = pd.read_csv(self.test_path)
        self.logger.info("Data loaded for ML model")

    def preprocess_data(self):
        self.X_train = self.train.drop(['Sales', 'Date'], axis=1)
        self.y_train = self.train['Sales']
        self.X_test = self.test.drop(['Sales', 'Date'], axis=1)
        self.y_test = self.test['Sales']
        self.logger.info("Data preprocessed for ML model")

    def build_model(self):
        self.pipeline = Pipeline([
            ('model', RandomForestRegressor(n_estimators=100))
        ])
        self.logger.info("Model pipeline created")

    def train_model(self):
        self.pipeline.fit(self.X_train, self.y_train)
        self.logger.info("Model trained")

    def evaluate_model(self):
        predictions = self.pipeline.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        self.logger.info(f"Model evaluation complete. MSE: {mse}")
        return mse

    def save_model(self, model_path):
        joblib.dump(self.pipeline, model_path)
        self.logger.info("Model saved")

    def run(self):
        self.load_data()
        self.preprocess_data()
        self.build_model()
        self.train_model()
        mse = self.evaluate_model()
        self.save_model(f"data/models/model-{pd.Timestamp.now().strftime('%Y-%m-%d-%H-%M-%S')}.pkl")
        return mse

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = MLModel('data/processed/train_processed.csv', 'data/processed/test_processed.csv')
    mse = model.run()
    print(f"Model MSE: {mse}")