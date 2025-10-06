import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pytest

from src import predict, metrics

metrics = metrics.Metrics()

def test_predict_data():
    # Sample input data
    X_sample = [
        [14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0],
        [13.2, 1.78, 2.14, 11.2, 100.0, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050.0],
        [12.37, 1.17, 1.92, 19.6, 78.0, 2.11, 2.0, 0.27, 1.04, 4.68, 1.12, 3.48, 510.0]
    ]
    # Expected output (class labels for the sample input)
    expected_output = [0, 0, 1]

    y_pred = list(predict.predict_data(X_sample))

    assert y_pred == expected_output, f"Expected {expected_output}, but got {y_pred}"

def test_get_model_accuracy():
    accuracy_value = metrics.get_model_accuracy()

    assert accuracy_value >= 0.90, f"Model accuracy {accuracy_value} is below the threshold"

def test_get_f1_score():
    f1_score_value = metrics.get_f1_score()

    assert f1_score_value >= 0.90, f"F1 score {f1_score_value} is below the threshold"

def test_get_precision():
    precision_value = metrics.get_precision()

    assert precision_value >= 0.90, f"Precision {precision_value} is below the threshold"

def test_get_recall():
    recall_value = metrics.get_recall()

    assert recall_value >= 0.90, f"Recall {recall_value} is below the threshold"

if __name__ == "__main__":
    pytest.main()