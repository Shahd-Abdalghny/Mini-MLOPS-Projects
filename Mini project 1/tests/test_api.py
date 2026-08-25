from unittest.mock import MagicMock
import numpy as np
import pytest
from prodml.predict import LaptopPricePredictor


def test_predict_one_with_mocked_session(sample_laptop):
    predictor = LaptopPricePredictor()
    
    mock_session = MagicMock()
    
    mock_session.run.return_value = [np.array([[np.log(50000)]])]
    predictor.session = mock_session

    price = predictor.predict_one(sample_laptop)

    
    assert mock_session.run.called
    assert price == pytest.approx(50000, rel=1e-3)