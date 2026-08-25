import pickle

import numpy as np
import onnxruntime as ort

from prodml.config import settings
from prodml.data import load_data
from prodml.features import build_features
from prodml.predict import _dataframe_to_onnx_inputs
import logging
from prodml.logging_conf import configure_logging

configure_logging()
logger = logging.getLogger(__name__)



def load_validation_rows(n: int = 500):
    df = load_data(settings.data_path)
    df = build_features(df)
    df = df.sample(n=min(n, len(df)), random_state=42)
    X = df[settings.FEATURE_COLUMNS]
    return X


def run_parity_test(onnx_path: str = "models/model.onnx", n: int = 500) -> None:
    with open(settings.model_path, "rb") as f:
        pickle_pipe = pickle.load(f)

    session = ort.InferenceSession(onnx_path)

    X = load_validation_rows(n)

   
    pred_pkl = pickle_pipe.predict(X)

    
    onnx_inputs = _dataframe_to_onnx_inputs(X)        
    onnx_output = session.run(None, onnx_inputs)
    pred_onnx = onnx_output[0].flatten()

    is_close = np.allclose(pred_pkl, pred_onnx, atol=1e-4)
    max_diff = np.max(np.abs(pred_pkl - pred_onnx))

    logger.info(f"Tested on {len(X)} rows")
    logger.info(f"Max absolute difference: {max_diff:.6f}")
    logger.info(f"Parity test passed: {is_close}")

    assert is_close, "ONNX predictions do not match pickle predictions!"


if __name__ == "__main__":
    run_parity_test()