import unittest
import json
from app.main import app

class TestModelServing(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home(self):
        result = self.app.get('/')
        self.assertEqual(result.status_code, 200)
        self.assertIn(b"Welcome to Rossmann Sales Forecasting API", result.data)

    def test_predict_ml(self):
        sample_data = {
            "Store": [1],
            "DayOfWeek": [5],
            "Customers": [555],
            "Open": [1],
            "Promo": [1],
            "StateHoliday": [0],
            "SchoolHoliday": [1],
            "Year": [2015],
            "Month": [7],
            "Day": [31],
            "WeekOfYear": [31]
        }
        result = self.app.post('/predict_ml', data=json.dumps(sample_data), content_type='application/json')
        self.assertEqual(result.status_code, 200)

    def test_predict_dl(self):
        sample_data = {
            "Store": [1],
            "DayOfWeek": [5],
            "Customers": [555],
            "Open": [1],
            "Promo": [1],
            "StateHoliday": [0],
            "SchoolHoliday": [1],
            "Year": [2015],
            "Month": [7],
            "Day": [31],
            "WeekOfYear": [31]
        }
        result = self.app.post('/predict_dl', data=json.dumps(sample_data), content_type='application/json')
        self.assertEqual(result.status_code, 200)

if __name__ == '__main__':
    unittest.main()
