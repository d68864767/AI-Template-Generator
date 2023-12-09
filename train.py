```python
# Import necessary libraries
from models.classification_template import train_model as train_classification_model
from models.regression_template import train_regression_model
from models.clustering_template import train_clustering_model
from models.dimensionality_reduction_template import train_dimensionality_reduction_model
from models.deep_learning_template import train_deep_learning_model
from data_loader import load_dataset, clean_dataset, split_dataset

def train(model_type, task):
    """
    Train the specified AI model on the given task.

    Parameters:
    model_type (str): The type of AI model to be trained.
    task (str): The task for which the model is being trained.
    """
    # Load and clean the dataset
    df = load_dataset(task)
    df_cleaned = clean_dataset(df)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = split_dataset(df_cleaned)

    # Train the model based on the model type
    if model_type == "Classification":
        model = train_classification_model(X_train, y_train)
    elif model_type == "Regression":
        model = train_regression_model(X_train, y_train)
    elif model_type == "Clustering":
        model = train_clustering_model(X_train)
    elif model_type == "Dimensionality Reduction":
        model = train_dimensionality_reduction_model(X_train)
    elif model_type == "Deep Learning":
        model = train_deep_learning_model(X_train, y_train)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Return the trained model
    return model

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python train.py <model_type> <task>")
        sys.exit(1)

    model_type = sys.argv[1]
    task = sys.argv[2]

    # Train the model
    model = train(model_type, task)
    print(f"Model trained for task: {task} using model type: {model_type}")
```
