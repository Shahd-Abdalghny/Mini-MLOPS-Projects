import numpy as np
import onnxruntime as ort
import pandas as pd

from prodml.config import settings
from prodml.features import build_features



def _dataframe_to_onnx_inputs(df: pd.DataFrame) -> dict:
    onnx_inputs = {}
    for col in settings.FEATURE_COLUMNS:
        onnx_name = settings.COLUMN_TO_ONNX_NAME[col]
        values = df[col].to_numpy()
        if col in settings.TEXT_COLUMNS:
            onnx_inputs[onnx_name] = values.reshape(-1, 1).astype(str)
        elif col in settings.INT_COLUMNS:
            onnx_inputs[onnx_name] = values.reshape(-1, 1).astype(np.int64)
        else:
            onnx_inputs[onnx_name] = values.reshape(-1, 1).astype(np.float32)
    return onnx_inputs


class LaptopPricePredictor:
    def __init__(self) -> None:
        self.session: ort.InferenceSession | None = None

    def load(self) -> None:
        self.session = ort.InferenceSession(settings.onnx_model_path)

    def predict_one(self, features: dict) -> float:
        if self.session is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        df = pd.DataFrame([features])
        df = build_features(df)
        onnx_inputs = _dataframe_to_onnx_inputs(df[settings.FEATURE_COLUMNS])

        output = self.session.run(None, onnx_inputs)
        log_price = output[0].flatten()[0]
        return float(np.exp(log_price))

    def predict_batch(self, rows: list[dict]) -> list[float]:
        if self.session is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        df = pd.DataFrame(rows)
        df = build_features(df)
        onnx_inputs = _dataframe_to_onnx_inputs(df[settings.FEATURE_COLUMNS])

        output = self.session.run(None, onnx_inputs)
        log_prices = output[0].flatten()
        return [float(p) for p in np.exp(log_prices)]