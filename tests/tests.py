import pytest
import numpy as np
from include.MLflow_model import eval_metrics

def test_eval_metrics_accuracy():
    # Test the math behind your model evaluation
    actual = np.array([10, 20, 30])
    pred = np.array([10, 20, 30])
    rmse, mae, r2 = eval_metrics(actual, pred)
    
    assert rmse == 0
    assert mae == 0
    assert r2 == 1.0

def test_prediction_output_type():
    # Example: Ensuring a mock prediction returns a float
    dummy_prediction = np.array([15.5])
    assert isinstance(dummy_prediction[0], (float, np.float64))