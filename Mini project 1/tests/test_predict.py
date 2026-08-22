import numpy as np
import pytest

from unittest.mock import MagicMock
from prodml.predict import LaptopPricePredictor


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = np.array([2.0])
    return model

@pytest.fixture
def mock_batch_model():
    model = MagicMock()
    model.predict.return_value = np.array([2.0, 3.0, 4.0])
    return model

def test_predictor_starts_without_model():
    predictor = LaptopPricePredictor()

    assert predictor.pipe is None


def test_predict_one(mock_model):
    predictor = LaptopPricePredictor()
    predictor.pipe = mock_model

    features = {
        "Unnamed: 0": 0,
        "ScreenResolution": "1920x1080",
        "Cpu": "Intel Core i5 8250U",
        "Gpu": "Intel UHD Graphics 620",
        "OpSys": "Windows 10",
        "Memory": "256GB SSD",
        "Ram": "8GB",
        "Weight": "1.5kg",
        "Inches": 15.6,
    }

    result = predictor.predict_one(features)

    assert result == pytest.approx(np.exp(2.0))

    mock_model.predict.assert_called_once()
    

def test_predict_batch(mock_batch_model):
    predictor = LaptopPricePredictor()
    predictor.pipe = mock_batch_model

    rows = [
        {
            "Unnamed: 0": 0,
            "ScreenResolution": "1920x1080",
            "Cpu": "Intel Core i5 8250U",
            "Gpu": "Intel UHD Graphics 620",
            "OpSys": "Windows 10",
            "Memory": "256GB SSD",
            "Ram": "8GB",
            "Weight": "1.5kg",
            "Inches": 15.6,
        },
        {
            "Unnamed: 0": 1,
            "ScreenResolution": "1366x768",
            "Cpu": "Intel Core i3 6006U",
            "Gpu": "Intel HD Graphics 520",
            "OpSys": "Windows 10",
            "Memory": "500GB HDD",
            "Ram": "4GB",
            "Weight": "2.0kg",
            "Inches": 15.6,
        },
        {
            "Unnamed: 0": 2,
            "ScreenResolution": "1920x1080",
            "Cpu": "Intel Core i7 7700HQ",
            "Gpu": "Nvidia GeForce GTX 1050",
            "OpSys": "Windows 10",
            "Memory": "256GB SSD + 1TB HDD",
            "Ram": "16GB",
            "Weight": "2.5kg",
            "Inches": 15.6,
        },
    ]

    result = predictor.predict_batch(rows)

    assert len(result) == 3
    assert result == pytest.approx(
        [np.exp(2.0), np.exp(3.0), np.exp(4.0)]
    )

    mock_batch_model.predict.assert_called_once()    