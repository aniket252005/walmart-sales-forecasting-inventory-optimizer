"""
prophet_model.py
----------------
Facebook Prophet forecasting model with automatic seasonality,
holiday support, and uncertainty intervals.

Usage
-----
    from prophet_model import ProphetForecaster
    model = ProphetForecaster(config)
    model.fit(train_df)          # train_df must have 'ds' and 'y' columns
    forecast = model.predict(periods=12)
"""

import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")


WALMART_HOLIDAYS = pd.DataFrame({
    "holiday": [
        "SuperBowl", "SuperBowl", "SuperBowl",
        "LaborDay", "LaborDay", "LaborDay",
        "Thanksgiving", "Thanksgiving", "Thanksgiving",
        "Christmas", "Christmas", "Christmas",
    ],
    "ds": pd.to_datetime([
        "2010-02-12", "2011-02-11", "2012-02-10",
        "2010-09-10", "2011-09-09", "2012-09-07",
        "2010-11-26", "2011-11-25", "2012-11-23",
        "2010-12-31", "2011-12-30", "2012-12-28",
    ]),
    "lower_window": [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    "upper_window": [ 1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2],
})


class ProphetForecaster:
    """Wrapper around Facebook Prophet with save/load support."""

    def __init__(self, config: dict, use_holidays: bool = True):
        self.config       = config["models"]["prophet"]
        self.output_dir   = Path(config["output"]["models_dir"])
        self.use_holidays = use_holidays
        self.model_       = None
        self.model_name   = "Prophet"

    # ------------------------------------------------------------------
    def fit(self, series: pd.Series) -> "ProphetForecaster":
        """
        Fit Prophet to a univariate time series.

        Parameters
        ----------
        series : pd.Series indexed by Date (datetime), values = Weekly_Sales.
        """
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Install prophet: pip install prophet")

        train_df = self._to_prophet_df(series)

        kwargs = dict(
            yearly_seasonality  = self.config["yearly_seasonality"],
            weekly_seasonality  = self.config["weekly_seasonality"],
            daily_seasonality   = self.config["daily_seasonality"],
            changepoint_prior_scale = self.config["changepoint_prior_scale"],
            seasonality_prior_scale = self.config["seasonality_prior_scale"],
            interval_width      = self.config["interval_width"],
        )
        if self.use_holidays:
            kwargs["holidays"] = WALMART_HOLIDAYS

        self.model_ = Prophet(**kwargs)
        self.model_.fit(train_df)
        print(f"[{self.model_name}] Fitted on {len(train_df)} observations.")
        return self

    # ------------------------------------------------------------------
    def predict(self, periods: int = 12, freq: str = "W") -> pd.DataFrame:
        """
        Generate future forecast.

        Parameters
        ----------
        periods : int  — number of future periods.
        freq    : str  — pandas frequency string ('W', 'D', 'MS', etc.)

        Returns
        -------
        pd.DataFrame with columns [ds, yhat, yhat_lower, yhat_upper].
        """
        self._check_fitted()
        future   = self.model_.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model_.predict(future)
        forecast["yhat"] = forecast["yhat"].clip(lower=0)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    # ------------------------------------------------------------------
    def predict_in_sample(self, test_series: pd.Series) -> np.ndarray:
        """Return predictions aligned to test_series dates."""
        self._check_fitted()
        test_df  = self._to_prophet_df(test_series)
        forecast = self.model_.predict(test_df)
        return forecast["yhat"].clip(lower=0).values

    # ------------------------------------------------------------------
    def plot_components(self) -> None:
        """Plot Prophet trend / seasonality components."""
        self._check_fitted()
        future   = self.model_.make_future_dataframe(periods=0, freq="W")
        forecast = self.model_.predict(future)
        self.model_.plot_components(forecast)

    # ------------------------------------------------------------------
    def save(self, filename: str = "prophet_model.pkl") -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        joblib.dump(self.model_, path)
        print(f"[{self.model_name}] Model saved → {path}")

    @classmethod
    def load(cls, config: dict, filename: str = "prophet_model.pkl") -> "ProphetForecaster":
        instance = cls(config)
        path     = Path(config["output"]["models_dir"]) / filename
        instance.model_ = joblib.load(path)
        print(f"[{cls.__name__}] Model loaded ← {path}")
        return instance

    # ------------------------------------------------------------------
    @staticmethod
    def _to_prophet_df(series: pd.Series) -> pd.DataFrame:
        """Convert a pd.Series (indexed by Date) to Prophet's ds/y format."""
        df = series.reset_index()
        df.columns = ["ds", "y"]
        df["ds"] = pd.to_datetime(df["ds"])
        return df

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

    sample = df[(df["Store"] == 1) & (df["Dept"] == 1)].set_index("Date")["Weekly_Sales"].sort_index()
    split  = int(len(sample) * 0.8)
    train, test = sample.iloc[:split], sample.iloc[split:]

    forecaster = ProphetForecaster(cfg, use_holidays=True)
    forecaster.fit(train)

    preds = forecaster.predict_in_sample(test)
    evaluate_model(test.values, preds, model_name="Prophet")
    forecaster.save()
