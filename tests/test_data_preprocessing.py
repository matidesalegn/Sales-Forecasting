import unittest
import pandas as pd
from scripts.data_preprocessing import DataPreprocessing

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        self.processor = DataPreprocessing('data/raw/train.csv', 'data/raw/test.csv', 'data/raw/store.csv')
        self.processor.load_data()

    def test_load_data(self):
        self.assertIsNotNone(self.processor.train)
        self.assertIsNotNone(self.processor.test)
        self.assertIsNotNone(self.processor.store)

    def test_merge_data(self):
        self.processor.merge_data()
        self.assertIn('StoreType', self.processor.train.columns)
        self.assertIn('StoreType', self.processor.test.columns)

    def test_preprocess(self):
        self.processor.preprocess()
        self.assertIn('Year', self.processor.train.columns)
        self.assertIn('Year', self.processor.test.columns)

if __name__ == '__main__':
    unittest.main()
