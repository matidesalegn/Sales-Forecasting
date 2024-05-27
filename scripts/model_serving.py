from flask import Flask, request, jsonify
import joblib
import tensorflow as tf
import pandas as pd

app = Flask(__name__)

# Load the models
ml_model = joblib.load('data/models/model-<timestamp>.pkl')
dl_model = tf.keras.models.load_model('data/models/lstm_model-<timestamp>.h5')

@app.route('/')
def home():
    return "Welcome to Rossmann Sales Forecasting API"

@app.route('/predict_ml', methods=['POST'])
def predict_ml():
    data = request.get_json(force=True)
    data_df = pd.DataFrame(data)
    prediction = ml_model.predict(data_df)
    return jsonify(prediction.tolist())

@app.route('/predict_dl', methods=['POST'])
def predict_dl():
    data = request.get_json(force=True)
    data_df = pd.DataFrame(data)
    scaled_data = scaler.transform(data_df)
    X = scaled_data.reshape((scaled_data.shape[0], scaled_data.shape[1], 1))
    prediction = dl_model.predict(X)
    return jsonify(prediction.tolist())

if __name__ == "__main__":
    app.run(debug=True)