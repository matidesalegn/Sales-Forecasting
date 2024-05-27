import unittest
from scripts.ml_model import MLModel

class TestMLModel(unittest.TestCase):
    def setUp(self):
        self.model = MLModel(train_path='data/processed/train_processed.csv', test_path='data/processed/test_processed.csv')
        self.model.load_data()
        self.model.preprocess_data()

    def test_load_data(self):
        self.assertIsNotNone(self.model.train)
        self.assertIsNotNone(self.model.test)

    def test_build_model(self):
        self.model.build_model()
        self.assertIsNotNone(self.model.pipeline)

    def test_train_model(self):
        self.model.build_model()
        self.model.train_model()
        self.assertTrue(self.model.pipeline)

if __name__ == '__main__':
    unittest.main()
