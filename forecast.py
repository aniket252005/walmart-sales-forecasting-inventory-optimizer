"""
forecast.py
-----------
Standalone forecasting script. Loads the best saved model and generates
a 30/60/90-day forecast for a given Store-Dept, then exports results.

Run
---
    python forecast.py --store 1 --dept 1 --horizon 90
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Make src importable
sys.path.insert(0, "src")

from data.data_loader     import load_config, load_processed_data
from models.prophet_model import ProphetForecaster
from models.arima_model   import ARIMAForecaster
from evaluation.metrics   import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Sales Forecast Generator")
    parser.add_argument("--store",   type=int, default=1,   help="Walmart store number")
    parser.add_argument("--dept",    type=int, default=1,   help="Department number")
    parser.add_argument("--horizon", type=int, default=90,  help="Forecast horizon in days")
    parser.add_argument("--model",   type=str, default="prophet",
                        choices=["prophet", "arima"],
                        help="Model to use for forecasting")
    return parser.parse_args()


def run_forecast(store: int, dept: int, horizon_days: int, model_type: str, config: dict) -> pd.DataFrame:
    df = load_processed_data(config)

    # Extract Store-Dept series
    mask   = (df["Store"] == store) & (df["Dept"] == dept)
    series = df[mask].set_index("Date")["Weekly_Sales"].sort_index()

    if series.empty:
        raise ValueError(f"No data found for Store={store}, Dept={dept}")

    horizon_weeks = max(1, horizon_days // 7)

    # Load or fit model
    out_dir = Path(config["output"]["models_dir"])
    if model_type == "prophet":
        fname = f"prophet_store{store}_dept{dept}.pkl"
        if (out_dir / fname).exists():
            forecaster = ProphetForecaster.load(config, filename=fname)
        else:
            forecaster = ProphetForecaster(config)
            forecaster.fit(series)
            forecaster.save(filename=fname)

        forecast_df = forecaster.predict(periods=horizon_weeks, freq="W")
        forecast_df = forecast_df.tail(horizon_weeks)
        forecast_df.columns = ["Date", "Forecast", "Lower_CI", "Upper_CI"]

    elif model_type == "arima":
        fname = f"arima_store{store}_dept{dept}.pkl"
        if (out_dir / fname).exists():
            forecaster = ARIMAForecaster.load(config, filename=fname)
        else:
            forecaster = ARIMAForecaster(config)
            forecaster.fit(series)
            forecaster.save(filename=fname)

        preds = forecaster.predict(steps=horizon_weeks)
        start = series.index[-1] + pd.DateOffset(weeks=1)
        dates = pd.date_range(start=start, periods=horizon_weeks, freq="W")
        forecast_df = pd.DataFrame({
            "Date":     dates,
            "Forecast": preds,
            "Lower_CI": preds * 0.85,
            "Upper_CI": preds * 1.15,
        })

    forecast_df["Store"] = store
    forecast_df["Dept"]  = dept
    return forecast_df


def save_forecast(forecast_df: pd.DataFrame, config: dict, store: int, dept: int) -> None:
    out_dir = Path(config["output"]["forecasts_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"forecast_store{store}_dept{dept}.csv"
    path  = out_dir / fname
    forecast_df.to_csv(path, index=False)
    print(f"[forecast] Saved → {path}")


if __name__ == "__main__":
    args   = parse_args()
    config = load_config()

    print(f"\nGenerating {args.horizon}-day {args.model.upper()} forecast "
          f"for Store={args.store}, Dept={args.dept}…\n")

    forecast = run_forecast(
        store       = args.store,
        dept        = args.dept,
        horizon_days= args.horizon,
        model_type  = args.model,
        config      = config,
    )

    print(forecast.to_string(index=False))
    save_forecast(forecast, config, args.store, args.dept)
