import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from prodml.config import settings
from prodml.features import build_features
import logging
from prodml.logging_conf import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

def build_pipeline(cat_cols: list[str]) -> Pipeline:
    step1 = ColumnTransformer(
        transformers=[
            ("col_tnf", OneHotEncoder(sparse_output=False, drop="first"), cat_cols)
        ],
        remainder="passthrough",
    )

    rf = RandomForestRegressor(
        n_estimators=settings.rf_n_estimators, random_state=settings.rf_random_state, max_samples=settings.rf_max_samples, max_features=settings.rf_max_features, max_depth=settings.rf_max_depth
    )
    gbdt = GradientBoostingRegressor(n_estimators=settings.gbdt_n_estimators, max_features=settings.gbdt_max_features)
    xgb = XGBRegressor(n_estimators=settings.xgb_n_estimators, learning_rate=settings.xgb_learning_rate, max_depth=settings.xgb_max_depth)
    et = ExtraTreesRegressor(
        n_estimators=settings.et_n_estimators, random_state=settings.et_random_state, max_samples=settings.et_max_samples, max_features=settings.et_max_features,
        max_depth=settings.et_max_depth, bootstrap=settings.et_bootstrap,
    )

    step2 = VotingRegressor(
        [("rf", rf), ("gbdt", gbdt), ("xgb", xgb), ("et", et)],
        weights=np.array([5, 1, 1, 1]),
    )

    return Pipeline([("step1", step1), ("step2", step2)])


def main() -> None:
    df = pd.read_csv(settings.data_path)
    df = build_features(df)

    X = df.drop(columns=["Price"])
    y = np.log(df["Price"])

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.test_size, random_state=settings.random_state
    )

    pipe = build_pipeline(cat_cols)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    logger.info(f"R2 Score: {r2:.4f}")
    logger.info(f"MAE (on log price): {mae:.4f}")

    with open(settings.model_path, "wb") as f:
        pickle.dump(pipe, f)
    logger.info(f"Model saved to {settings.model_path}")


if __name__ == "__main__":
    main()