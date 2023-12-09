```python
def get_model_template_path(model_type):
    """
    Get the file path for the model template based on the model type.

    Parameters:
    model_type (str): The type of AI model.

    Returns:
    str: The file path of the model template.
    """
    model_templates = {
        "Classification": "models/classification_template.py",
        "Regression": "models/regression_template.py",
        "Clustering": "models/clustering_template.py",
        "Dimensionality Reduction": "models/dimensionality_reduction_template.py",
        "Deep Learning": "models/deep_learning_template.py"
    }

    if model_type in model_templates:
        return model_templates[model_type]
    else:
        raise ValueError(f"No template available for the model type: {model_type}")

def load_model_template(model_type):
    """
    Load the model template based on the model type.

    Parameters:
    model_type (str): The type of AI model.

    Returns:
    str: The content of the model template file.
    """
    template_path = get_model_template_path(model_type)
    with open(template_path, 'r') as file:
        template_content = file.read()
    return template_content

def print_model_template(model_type, task):
    """
    Print the model template for the given model type and task.

    Parameters:
    model_type (str): The type of AI model.
    task (str): The task the AI model is supposed to perform.
    """
    template_content = load_model_template(model_type)
    print(f"// Model Type: {model_type}")
    print(f"// Task: {task}\n")
    print(template_content)

def validate_input(model_type, task):
    """
    Validate the input model type and task.

    Parameters:
    model_type (str): The type of AI model.
    task (str): The task the AI model is supposed to perform.

    Raises:
    ValueError: If the model type or task is not supported.
    """
    supported_model_types = ["Classification", "Regression", "Clustering", "Dimensionality Reduction", "Deep Learning"]
    supported_tasks = ["Spam Filtering", "House Price Prediction", "Customer Segmentation", "Feature Selection", "Image Recognition"]

    if model_type not in supported_model_types:
        raise ValueError(f"Unsupported model type: {model_type}")

    if task not in supported_tasks:
        raise ValueError(f"Unsupported task: {task}")
```
