```python
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(task):
    """
    Load the dataset based on the specified task.
    This is a placeholder function and should be modified to load actual datasets.
    
    Parameters:
    task (str): The task for which the dataset needs to be loaded.

    Returns:
    DataFrame: The loaded dataset.
    """
    # Placeholder for actual dataset loading logic
    # This should be replaced with actual code to load the dataset for the given task
    datasets = {
        "Spam Filtering": "path/to/spam_filtering_dataset.csv",
        "House Price Prediction": "path/to/house_price_dataset.csv",
        "Customer Segmentation": "path/to/customer_segmentation_dataset.csv",
        "Feature Selection": "path/to/feature_selection_dataset.csv",
        "Image Recognition": "path/to/image_recognition_dataset.csv"
    }

    if task in datasets:
        dataset_path = datasets[task]
        return pd.read_csv(dataset_path)
    else:
        raise ValueError(f"No dataset available for the task: {task}")

def clean_dataset(df):
    """
    Clean the dataset by handling missing values, encoding categorical variables, etc.
    This is a placeholder function and should be modified based on the dataset and task.

    Parameters:
    df (DataFrame): The dataset to be cleaned.

    Returns:
    DataFrame: The cleaned dataset.
    """
    # Placeholder for actual data cleaning logic
    # This should be replaced with actual code to clean the dataset
    # For example, you might fill missing values, encode categorical variables, etc.
    return df  # Assuming df is the cleaned dataset

def split_dataset(df, test_size=0.2, random_state=None):
    """
    Split the dataset into training and testing sets.

    Parameters:
    df (DataFrame): The dataset to be split.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
    tuple: The training and testing sets as DataFrames.
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df
```
