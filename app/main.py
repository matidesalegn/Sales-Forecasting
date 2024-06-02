# app/main.py
from flask import Blueprint, render_template, request
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler

main = Blueprint('main', __name__)

# Load the trained models
rf_model = joblib.load('../models/model-2024-06-01-10-36-59.pkl')
lstm_model = tf.keras.models.load_model('../models/lstm_model.h5')

# Load scalers (if used during preprocessing)
scaler = joblib.load('../models/scaler.pkl')  # Ensure to save scaler during training

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    date_input = request.form.get('date')
    date = pd.to_datetime(date_input)

    # Feature engineering for the input date
    features = {
        'Year': date.year,
        'Month': date.month,
        'Day': date.day,
        'WeekOfYear': date.isocalendar()[1],
        'Weekday': date.weekday(),
        'IsWeekend': date.weekday() >= 5,
        'MonthStart': date.day <= 10,
        'MonthMid': 10 < date.day <= 20,
        'MonthEnd': date.day > 20,
        # Add other relevant features used during model training
    }
    features_df = pd.DataFrame([features])

    # Scale features if necessary
    scaled_features = scaler.transform(features_df)

    # Random Forest Prediction
    rf_prediction = rf_model.predict(scaled_features)[0]

    # LSTM Prediction
    lstm_features = np.array(scaled_features).reshape((1, scaled_features.shape[1], 1))
    lstm_prediction = lstm_model.predict(lstm_features)[0][0]

    return render_template('result.html', date=date_input, rf_prediction=rf_prediction, lstm_prediction=lstm_prediction)