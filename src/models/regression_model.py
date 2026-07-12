"""
regression_model.py
-------------------
Baseline linear regression + tree-based models (Random Forest, XGBoost)
for multi-step sales forecasting using engineered features.

Models exposed:
  LinearRegressionModel
  RandomForestModel
  XGBoostModel
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _BaseModel:
    """Shared fit / predict / save / load interface."""

    def __init__(self, config: dict, model_name: str):
        self.config     = config
        self.output_dir = Path(config["output"]["models_dir"])
        self.model_name = model_name
        self.pipeline_  = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "_BaseModel":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        preds = self.pipeline_.predict(X)
        return np.maximum(preds, 0)   # sales cannot be negative

    def feature_importances(self, feature_names: list) -> pd.Series:
        """Return feature importances if the underlying model supports it."""
        self._check_fitted()
        estimator = self.pipeline_.named_steps.get("model")
        if hasattr(estimator, "feature_importances_"):
            return pd.Series(
                estimator.feature_importances_,
                index=feature_names,
            ).sort_values(ascending=False)
        raise AttributeError(f"{self.model_name} does not expose feature_importances_.")

    def save(self, filename: str = None) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fname = filename or f"{self.model_name.lower().replace(' ', '_')}.pkl"
        path  = self.output_dir / fname
        joblib.dump(self.pipeline_, path)
        print(f"[{self.model_name}] Saved → {path}")

    @classmethod
    def _load_pipeline(cls, config: dict, model_name: str, filename: str):
        path = Path(config["output"]["models_dir"]) / filename
        instance = cls.__new__(cls)
        instance.config     = config
        instance.output_dir = Path(config["output"]["models_dir"])
        instance.model_name = model_name
        instance.pipeline_  = joblib.load(path)
        print(f"[{model_name}] Loaded ← {path}")
        return instance

    def _check_fitted(self) -> None:
        if self.pipeline_ is None:
            raise RuntimeError(f"{self.model_name}: not fitted. Call .fit() first.")


# ---------------------------------------------------------------------------
# Linear Regression (baseline)
# ---------------------------------------------------------------------------

class LinearRegressionModel(_BaseModel):
    """Ridge regression with feature scaling — good interpretable baseline."""

    def __init__(self, config: dict):
        super().__init__(config, "Linear Regression")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LinearRegressionModel":
        self.pipeline_ = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0)),
        ])
        self.pipeline_.fit(X_train, y_train)
        print(f"[{self.model_name}] Fitted on {len(X_train):,} samples.")
        return self

    @classmethod
    def load(cls, config: dict, filename: str = "linear_regression.pkl") -> "LinearRegressionModel":
        return cls._load_pipeline(config, "Linear Regression", filename)


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

class RandomForestModel(_BaseModel):
    """Random Forest Regressor — handles non-linearity, gives feature importances."""

    def __init__(self, config: dict):
        super().__init__(config, "Random Forest")
        self.rf_config = config["models"]["random_forest"]

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RandomForestModel":
        rf = RandomForestRegressor(
            n_estimators   = self.rf_config["n_estimators"],
            max_depth      = self.rf_config["max_depth"],
            min_samples_split = self.rf_config["min_samples_split"],
            random_state   = self.config["models"]["random_state"],
            n_jobs         = self.rf_config["n_jobs"],
        )
        self.pipeline_ = Pipeline([("model", rf)])
        self.pipeline_.fit(X_train, y_train)
        print(
            f"[{self.model_name}] Fitted on {len(X_train):,} samples | "
            f"n_estimators={self.rf_config['n_estimators']}"
        )
        return self

    @classmethod
    def load(cls, config: dict, filename: str = "random_forest.pkl") -> "RandomForestModel":
        return cls._load_pipeline(config, "Random Forest", filename)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class XGBoostModel(_BaseModel):
    """XGBoost Regressor — typically best-in-class for tabular time-series features."""

    def __init__(self, config: dict):
        super().__init__(config, "XGBoost")
        self.xgb_config = config["models"]["xgboost"]

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "XGBoostModel":
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("Install XGBoost: pip install xgboost")

        xgb = XGBRegressor(
            n_estimators     = self.xgb_config["n_estimators"],
            learning_rate    = self.xgb_config["learning_rate"],
            max_depth        = self.xgb_config["max_depth"],
            subsample        = self.xgb_config["subsample"],
            colsample_bytree = self.xgb_config["colsample_bytree"],
            random_state     = self.config["models"]["random_state"],
            verbosity        = 0,
            n_jobs           = -1,
        )
        self.pipeline_ = Pipeline([("model", xgb)])
        self.pipeline_.fit(X_train, y_train)
        print(f"[{self.model_name}] Fitted on {len(X_train):,} samples.")
        return self

    @classmethod
    def load(cls, config: dict, filename: str = "xgboost.pkl") -> "XGBoostModel":
        return cls._load_pipeline(config, "XGBoost", filename)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.data_loader import load_config, load_features_data
    from evaluation.metrics import evaluate_model, compare_models
    from evaluation.backtesting import simple_time_split, get_feature_columns

    cfg = load_config()
    df  = load_features_data(cfg)

    # Use store 1, dept 1
    sample = df[(df["Store"] == 1) & (df["Dept"] == 1)].copy()
    feature_cols = get_feature_columns(sample)
    sample = sample.dropna(subset=feature_cols)

    X_train, X_test, y_train, y_test = simple_time_split(
        sample, "Weekly_Sales", feature_cols
    )

    results = []
    for ModelCls in [LinearRegressionModel, RandomForestModel, XGBoostModel]:
        m = ModelCls(cfg)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        results.append(evaluate_model(y_test, preds, model_name=m.model_name))
        m.save()

    compare_models(results)
