# src/prodml/data.py
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    return df