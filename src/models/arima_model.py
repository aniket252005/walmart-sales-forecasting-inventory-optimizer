"""
arima_model.py
--------------
ARIMA / SARIMA forecasting model using pmdarima's auto_arima for
automatic parameter selection.

Usage
-----
    from arima_model import ARIMAForecaster
    model = ARIMAForecaster(config)
    model.fit(train_series)
    preds = model.predict(steps=12)
"""

import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")


class ARIMAForecaster:
    """Wrapper around pmdarima auto_arima with save/load support."""

    def __init__(self, config: dict):
        self.config     = config["models"]["arima"]
        self.output_dir = Path(config["output"]["models_dir"])
        self.model_     = None
        self.model_name = "ARIMA"

    # ------------------------------------------------------------------
    def fit(self, series: pd.Series) -> "ARIMAForecaster":
        """
        Fit auto_arima to a univariate time series.

        Parameters
        ----------
        series : pd.Series
            Weekly sales indexed by Date, sorted chronologically.
        """
        try:
            from pmdarima import auto_arima
        except ImportError:
            raise ImportError("Install pmdarima: pip install pmdarima")

        print(f"[{self.model_name}] Fitting auto_arima (this may take a moment)…")
        self.model_ = auto_arima(
            series,
            max_p=self.config["max_p"],
            max_d=self.config["max_d"],
            max_q=self.config["max_q"],
            seasonal=self.config["seasonal"],
            m=self.config["m"],
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )
        order         = self.model_.order
        seasonal_order = getattr(self.model_, "seasonal_order", None)
        print(f"[{self.model_name}] Best order: {order}  seasonal: {seasonal_order}")
        return self

    # ------------------------------------------------------------------
    def predict(self, steps: int = 12) -> np.ndarray:
        """
        Generate out-of-sample forecasts.

        Parameters
        ----------
        steps : int
            Number of future periods to predict.

        Returns
        -------
        np.ndarray of length `steps`.
        """
        self._check_fitted()
        forecast = self.model_.predict(n_periods=steps)
        return np.maximum(forecast, 0)   # sales cannot be negative

    # ------------------------------------------------------------------
    def predict_in_sample(self, test_series: pd.Series) -> np.ndarray:
        """
        Re-fit model on training data then predict for the length of test_series.
        Used for evaluation against known actuals.
        """
        self._check_fitted()
        n = len(test_series)
        return self.predict(steps=n)

    # ------------------------------------------------------------------
    def summary(self) -> None:
        """Print the fitted model summary."""
        self._check_fitted()
        print(self.model_.summary())

    # ------------------------------------------------------------------
    def save(self, filename: str = "arima_model.pkl") -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        joblib.dump(self.model_, path)
        print(f"[{self.model_name}] Model saved → {path}")

    @classmethod
    def load(cls, config: dict, filename: str = "arima_model.pkl") -> "ARIMAForecaster":
        instance = cls(config)
        path = Path(config["output"]["models_dir"]) / filename
        instance.model_ = joblib.load(path)
        print(f"[{cls.__name__}] Model loaded ← {path}")
        return instance

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.data_loader import load_config, load_processed_data
    from evaluation.metrics import evaluate_model

    cfg    = load_config()
    df     = load_processed_data(cfg)
    if df.index.name == "Date":
        df = df.reset_index()

    # Single store-dept series
    sample = df[(df["Store"] == 1) & (df["Dept"] == 1)].set_index("Date")["Weekly_Sales"].sort_index()

    split  = int(len(sample) * 0.8)
    train, test = sample.iloc[:split], sample.iloc[split:]

    forecaster = ARIMAForecaster(cfg)
    forecaster.fit(train)

    preds = forecaster.predict(steps=len(test))
    evaluate_model(test.values, preds, model_name="ARIMA")
    forecaster.save()
