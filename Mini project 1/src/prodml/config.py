from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Configuration settings for the application."""

    data_path: str = "data/raw/laptop_data.csv"
    model_path: str = "models/model.pkl"
    test_size: float = 0.15
    random_state: int = 2
    random_seed: int = 42
    #n_estimators=350, random_state=3, max_samples=0.5, max_features=0.75, max_depth=15
    rf_n_estimators: int = 350
    rf_random_state: int = 3
    rf_max_samples: float = 0.5
    rf_max_features: float = 0.75
    rf_max_depth: int = 15
    # n_estimators=100, max_features=0.5
    gbdt_n_estimators: int = 100
    gbdt_max_features: float = 0.5
    # n_estimators=25, learning_rate=0.3, max_depth=5
    xgb_n_estimators: int = 25
    xgb_learning_rate: float = 0.3
    xgb_max_depth: int = 5
    # n_estimators=100, random_state=3, max_samples=0.5, max_features=0.75, max_depth=10, bootstrap=True
    et_n_estimators: int = 100
    et_random_state: int = 3
    et_max_samples: float = 0.5
    et_max_features: float = 0.75
    et_max_depth: int = 10
    et_bootstrap: bool = True

settings = Settings()    



