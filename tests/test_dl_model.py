import unittest
from scripts.dl_model import DLModel

class TestDLModel(unittest.TestCase):
    def setUp(self):
        self.model = DLModel(train_path='data/processed/train_processed.csv', test_path='data/processed/test_processed.csv')
        self.model.load_data()
        self.model.preprocess_data()

    def test_load_data(self):
        self.assertIsNotNone(self.model.train)
        self.assertIsNotNone(self.model.test)

    def test_build_model(self):
        self.model.build_model(n_steps=5)
        self.assertIsNotNone(self.model.model)

    def test_train_model(self):
        self.model.build_model(n_steps=5)
        self.model.train_model(n_steps=5)
        self.assertTrue(self.model.model)

if __name__ == '__main__':
    unittest.main()