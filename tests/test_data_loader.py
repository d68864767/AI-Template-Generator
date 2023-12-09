```python
import unittest
from unittest.mock import patch
import pandas as pd
from data_loader import load_dataset, clean_dataset, split_dataset

class TestDataLoader(unittest.TestCase):

    @patch('data_loader.pd.read_csv')
    def test_load_dataset(self, mock_read_csv):
        # Mocking the pandas read_csv function to return a DataFrame
        mock_read_csv.return_value = pd.DataFrame()

        # Test for known tasks
        for task in ["Spam Filtering", "House Price Prediction", "Customer Segmentation", "Feature Selection", "Image Recognition"]:
            df = load_dataset(task)
            self.assertIsInstance(df, pd.DataFrame)

        # Test for unknown task
        with self.assertRaises(ValueError):
            load_dataset("Unknown Task")

    def test_clean_dataset(self):
        # Create a mock DataFrame
        df = pd.DataFrame({
            'col1': [1, 2, 3, None],
            'col2': ['a', 'b', None, 'd']
        })

        # Clean the dataset
        cleaned_df = clean_dataset(df)

        # Check if the cleaned dataset is still a DataFrame
        self.assertIsInstance(cleaned_df, pd.DataFrame)

        # Placeholder test, should be replaced with actual cleaning checks
        # For example, check if missing values have been filled
        # self.assertEqual(cleaned_df.isnull().sum().sum(), 0)

    def test_split_dataset(self):
        # Create a mock DataFrame
        df = pd.DataFrame({
            'feature': range(10),
            'target': range(10)
        })

        # Split the dataset
        train_df, test_df = split_dataset(df, test_size=0.2)

        # Check if the training and testing sets are of the correct size
        self.assertEqual(len(train_df), 8)
        self.assertEqual(len(test_df), 2)

        # Check if the training and testing sets are DataFrames
        self.assertIsInstance(train_df, pd.DataFrame)
        self.assertIsInstance(test_df, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()
```
