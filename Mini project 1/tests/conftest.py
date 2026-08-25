import pytest
from prodml.predict import LaptopPricePredictor


@pytest.fixture
def sample_laptop() -> dict:
    return {
        "Company": "Dell",
        "TypeName": "Notebook",
        "ScreenResolution": "Full HD 1920x1080",
        "Cpu": "Intel Core i5 7200U 2.5GHz",
        "Ram": "8GB",
        "Memory": "256GB SSD",
        "Gpu": "Intel HD Graphics 620",
        "OpSys": "Windows 10",
        "Weight": "1.5kg",
        "Inches": 15.6,
    }


@pytest.fixture(scope="session")
def trained_predictor() -> LaptopPricePredictor:
    predictor = LaptopPricePredictor()
    predictor.load()
    return predictor

