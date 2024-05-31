import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import joblib
import logging
import os
from datetime import datetime

class MLModel:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.pipeline = None

    def load_data(self):
        dtype_spec = {
            'StateHoliday': str,
            'StoreType': str,
            'Assortment': str,
            'PromoInterval': str
        }
        self.train = pd.read_csv(self.train_path, dtype=dtype_spec)
        self.test = pd.read_csv(self.test_path, dtype=dtype_spec)
        self.X_train = self.train.drop(columns=['Sales'])
        self.y_train = self.train['Sales']
        self.X_test = self.test.copy()  # Sales column not available in test data
        self.logger.info("Data loaded successfully")

    def preprocess(self):
        # Identify numerical and categorical columns
        numerical_cols = self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = self.X_train.select_dtypes(include=['object', 'bool']).columns.tolist()
        
        # Ensure all categorical columns are of string type
        self.X_train[categorical_cols] = self.X_train[categorical_cols].astype(str)
        self.X_test[categorical_cols] = self.X_test[categorical_cols].astype(str)
        
        # Define preprocessing steps
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('model', RandomForestRegressor(n_estimators=100, random_state=42))])
        self.logger.info("Preprocessing pipeline created")

    def build_model(self):
        self.pipeline.fit(self.X_train, self.y_train)
        self.logger.info("Model trained successfully")

    def evaluate_model(self):
        predictions = self.pipeline.predict(self.X_train)
        mse = mean_squared_error(self.y_train, predictions)
        rmse = mean_squared_error(self.y_train, predictions, squared=False)
        self.logger.info(f"Model evaluation complete. RMSE: {rmse}")
        return rmse

    def save_model(self):
        model_dir = 'data/models'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, f"model-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pkl")
        joblib.dump(self.pipeline, model_path)
        self.logger.info(f"Model saved at {model_path}")

    def run(self):
        self.load_data()
        self.preprocess()
        self.build_model()
        rmse = self.evaluate_model()
        self.save_model()
        return rmse

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = MLModel('../data/processed/train_processed.csv', '../data/processed/test_processed.csv')
    rmse = model.run()
    print(f"Model RMSE: {rmse}")