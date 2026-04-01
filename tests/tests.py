import pytest
import numpy as np
# This import works because of the PYTHONPATH we set in the YAML
from MLflow_model import eval_metrics 

def test_eval_metrics_logic():
    # Test with perfect prediction
    actual = np.array([10.0, 20.0, 30.0])
    pred = np.array([10.0, 20.0, 30.0])
    rmse, mae, r2 = eval_metrics(actual, pred)
    
    assert rmse == 0.0
    assert mae == 0.0
    assert r2 == 1.0

def test_eval_metrics_error():
    # Test with known error
    actual = np.array([10.0])
    pred = np.array([12.0])
    rmse, mae, r2 = eval_metrics(actual, pred)
    
    assert mae == 2.0
    assert rmse == 2.0