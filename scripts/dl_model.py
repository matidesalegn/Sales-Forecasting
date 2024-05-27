import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import logging

class DLModel:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        self.train = pd.read_csv(self.train_path)
        self.test = pd.read_csv(self.test_path)
        self.logger.info("Data loaded for DL model")

    def preprocess_data(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.train['Sales'] = self.scaler.fit_transform(self.train[['Sales']])
        self.test['Sales'] = self.scaler.transform(self.test[['Sales']])
        self.logger.info("Data preprocessed for DL model")

    def prepare_lstm_data(self, data, n_steps):
        X, y = [], []
        for i in range(len(data)):
            end_ix = i + n_steps
            if end_ix > len(data)-1:
                break
            seq_x, seq_y = data[i:end_ix], data[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def build_model(self, n_steps):
        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, activation='relu', input_shape=(n_steps, 1)),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.logger.info("LSTM model built")

    def train_model(self, n_steps):
        X_train, y_train = self.prepare_lstm_data(self.train['Sales'].values, n_steps)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        self.model.fit(X_train, y_train, epochs=50, verbose=1)
        self.logger.info("LSTM model trained")

    def evaluate_model(self, n_steps):
        X_test, y_test = self.prepare_lstm_data(self.test['Sales'].values, n_steps)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        loss = self.model.evaluate(X_test, y_test, verbose=0)
        self.logger.info(f"LSTM model evaluation complete. Loss: {loss}")
        return loss

    def save_model(self, model_path):
        self.model.save(model_path)
        self.logger.info("LSTM model saved")

    def run(self, n_steps=5):
        self.load_data()
        self.preprocess_data()
        self.build_model(n_steps)
        self.train_model(n_steps)
        loss = self.evaluate_model(n_steps)
        self.save_model(f"data/models/lstm_model-{pd.Timestamp.now().strftime('%Y-%m-%d-%H-%M-%S')}.h5")
        return loss

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = DLModel('data/processed/train_processed.csv', 'data/processed/test_processed.csv')
    loss = model.run()
    print(f"Model Loss: {loss}")
