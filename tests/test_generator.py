```python
import unittest
from generator import generate_template

class TestGenerator(unittest.TestCase):

    def test_classification_template(self):
        model_type = 'Classification'
        task = 'Spam Filtering'
        expected_output = (
            "// Data Preprocessing\n"
            "// Load Dataset\n"
            "// Clean Dataset\n"
            "// Split into Training and Test Sets\n\n"
            "// Model Definition\n"
            "// Define Classification Model\n\n"
            "// Model Training\n"
            "// Train Model on Training Data\n\n"
            "// Model Evaluation\n"
            "// Evaluate Model on Test Data\n"
        )
        self.assertEqual(generate_template(model_type, task), expected_output)

    def test_regression_template(self):
        model_type = 'Regression'
        task = 'House Price Prediction'
        expected_output = (
            "// Data Preprocessing\n"
            "// Load Dataset\n"
            "// Clean Dataset\n"
            "// Split into Training and Test Sets\n\n"
            "// Model Definition\n"
            "// Define Regression Model\n\n"
            "// Model Training\n"
            "// Train Model on Training Data\n\n"
            "// Model Evaluation\n"
            "// Evaluate Model on Test Data\n"
        )
        self.assertEqual(generate_template(model_type, task), expected_output)

    def test_clustering_template(self):
        model_type = 'Clustering'
        task = 'Customer Segmentation'
        expected_output = (
            "// Data Preprocessing\n"
            "// Load Dataset\n"
            "// Clean Dataset\n"
            "// Feature Engineering\n\n"
            "// Model Definition\n"
            "// Define Clustering Model\n\n"
            "// Model Training\n"
            "// Train Model on Data\n\n"
            "// Model Evaluation\n"
            "// Evaluate Clustering Performance\n"
        )
        self.assertEqual(generate_template(model_type, task), expected_output)

    def test_dimensionality_reduction_template(self):
        model_type = 'Dimensionality Reduction'
        task = 'Feature Selection'
        expected_output = (
            "// Data Preprocessing\n"
            "// Load Dataset\n"
            "// Clean Dataset\n\n"
            "// Model Definition\n"
            "// Define Dimensionality Reduction Model\n\n"
            "// Model Training\n"
            "// Train Model on Data\n\n"
            "// Model Evaluation\n"
            "// Evaluate Model Performance\n"
        )
        self.assertEqual(generate_template(model_type, task), expected_output)

    def test_deep_learning_template(self):
        model_type = 'Deep Learning'
        task = 'Image Recognition'
        expected_output = (
            "// Data Preprocessing\n"
            "// Load Dataset\n"
            "// Clean Dataset\n"
            "// Augment Data\n\n"
            "// Model Definition\n"
            "// Define Deep Learning Model\n\n"
            "// Model Training\n"
            "// Train Model on Training Data\n\n"
            "// Model Evaluation\n"
            "// Evaluate Model on Test Data\n"
        )
        self.assertEqual(generate_template(model_type, task), expected_output)

if __name__ == '__main__':
    unittest.main()
```
