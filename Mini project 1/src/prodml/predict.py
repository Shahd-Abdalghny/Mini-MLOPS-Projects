import pickle
import numpy as np
import pandas as pd
from prodml.config import settings
from prodml.features import build_features

class LaptopPricePredictor:
    def __init__(self):
        self.pipe = None

    def load(self) -> None:
        with open(settings.model_path, "rb") as f:
            self.pipe = pickle.load(f)

    def predict_one(self, features: dict) -> float:
        df = pd.DataFrame([features])
        df = build_features(df)
        log_price = self.pipe.predict(df)[0]
        return float(np.exp(log_price))

    def predict_batch(self, rows: list[dict]) -> list[float]:
        df = pd.DataFrame(rows)
        df = build_features(df)
        log_prices = self.pipe.predict(df)
        return list(np.exp(log_prices))