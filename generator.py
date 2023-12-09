```python
# Import necessary modules
from models.classification_template import preprocess_data as preprocess_classification, define_classification_model, train_model as train_classification_model, evaluate_model as evaluate_classification_model
from models.regression_template import preprocess_data as preprocess_regression, define_regression_model, train_regression_model, evaluate_regression_model
from models.clustering_template import preprocess_data as preprocess_clustering, define_clustering_model, train_clustering_model, evaluate_clustering_model
from models.dimensionality_reduction_template import preprocess_data as preprocess_dimensionality_reduction, define_dimensionality_reduction_model, train_dimensionality_reduction_model, evaluate_dimensionality_reduction_model
from models.deep_learning_template import preprocess_data as preprocess_deep_learning, define_deep_learning_model, train_deep_learning_model, evaluate_deep_learning_model

# Define a dictionary to map model types to their respective functions
model_functions = {
    'Classification': {
        'preprocess': preprocess_classification,
        'define_model': define_classification_model,
        'train_model': train_classification_model,
        'evaluate_model': evaluate_classification_model
    },
    'Regression': {
        'preprocess': preprocess_regression,
        'define_model': define_regression_model,
        'train_model': train_regression_model,
        'evaluate_model': evaluate_regression_model
    },
    'Clustering': {
        'preprocess': preprocess_clustering,
        'define_model': define_clustering_model,
        'train_model': train_clustering_model,
        'evaluate_model': evaluate_clustering_model
    },
    'Dimensionality Reduction': {
        'preprocess': preprocess_dimensionality_reduction,
        'define_model': define_dimensionality_reduction_model,
        'train_model': train_dimensionality_reduction_model,
        'evaluate_model': evaluate_dimensionality_reduction_model
    },
    'Deep Learning': {
        'preprocess': preprocess_deep_learning,
        'define_model': define_deep_learning_model,
        'train_model': train_deep_learning_model,
        'evaluate_model': evaluate_deep_learning_model
    }
}

def generate_template(model_type, task):
    """
    Generate the template for the specified AI model and task.

    Parameters:
    model_type (str): The type of AI model.
    task (str): The task the AI model is supposed to perform.

    Returns:
    str: The generated template.
    """
    # Check if the model type is valid
    if model_type not in model_functions:
        raise ValueError(f"Model type '{model_type}' is not supported.")

    # Retrieve the functions for the specified model type
    functions = model_functions[model_type]

    # Generate the template
    template = f"""
# Data Preprocessing
# Load Dataset
# Clean Dataset
# Split into Training and Test Sets

# Model Definition
# Define {model_type} Model

# Model Training
# Train Model on Training Data

# Model Evaluation
# Evaluate Model on Test Data
"""

    return template

if __name__ == "__main__":
    import sys

    # Read model type and task from input
    model_type = sys.argv[1]
    task = sys.argv[2]

    # Generate the template
    try:
        template = generate_template(model_type, task)
        print(template)
    except ValueError as e:
        print(e)
```
