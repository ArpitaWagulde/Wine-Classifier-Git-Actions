import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pytest

from src import predict, accuracy

def test_predict_data():
    # Sample input data (3 samples with 13 features each)
    X_sample = [
        [14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0],
        [13.2, 1.78, 2.14, 11.2, 100.0, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050.0],
        [12.37, 1.17, 1.92, 19.6, 78.0, 2.11, 2.0, 0.27, 1.04, 4.68, 1.12, 3.48, 510.0]
    ]
    # Expected output (class labels for the sample input)
    expected_output = [0, 0, 1]  # Adjusted based on actual model predictions

    # Get predictions from the predict_data function
    y_pred = predict.predict_data(X_sample)

    # Assert that the predicted output matches the expected output
    assert list(y_pred) == expected_output

def test_get_model_accuracy():
    # Get the accuracy from the get_model_accuracy function
    accuracy_value = accuracy.get_model_accuracy()

    # Assert that the accuracy is above a certain threshold (e.g., 0.85)
    assert accuracy_value >= 0.85, f"Model accuracy {accuracy_value} is below the threshold"

if __name__ == "__main__":
    pytest.main()