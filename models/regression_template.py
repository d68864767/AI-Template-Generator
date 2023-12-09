```python
# Import necessary libraries
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from data_loader import load_dataset, clean_dataset, split_dataset

def preprocess_data(df):
    """
    Preprocess the data by scaling numerical features.

    Parameters:
    df (DataFrame): The dataset to be preprocessed.

    Returns:
    DataFrame: The preprocessed dataset.
    """
    scaler = StandardScaler()
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df

def define_regression_model():
    """
    Define the regression model.

    Returns:
    model: The defined regression model.
    """
    model = LinearRegression()
    return model

def train_regression_model(model, X_train, y_train):
    """
    Train the regression model on the training data.

    Parameters:
    model: The regression model to be trained.
    X_train (DataFrame): The training data features.
    y_train (Series): The training data target.

    Returns:
    model: The trained regression model.
    """
    model.fit(X_train, y_train)
    return model

def evaluate_regression_model(model, X_test, y_test):
    """
    Evaluate the regression model on the test data.

    Parameters:
    model: The regression model to be evaluated.
    X_test (DataFrame): The test data features.
    y_test (Series): The test data target.

    Returns:
    dict: A dictionary containing the evaluation metrics.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return {
        'Mean Squared Error': mse,
        'R^2 Score': r2
    }

# Example usage:
if __name__ == "__main__":
    # Load and preprocess the dataset
    df = load_dataset("House Price Prediction")
    df = clean_dataset(df)
    df = preprocess_data(df)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = split_dataset(df)

    # Define the model
    model = define_regression_model()

    # Train the model
    model = train_regression_model(model, X_train, y_train)

    # Evaluate the model
    evaluation_metrics = evaluate_regression_model(model, X_test, y_test)
    print(evaluation_metrics)
```
