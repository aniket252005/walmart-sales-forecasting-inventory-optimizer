"""
helpers.py
----------
Miscellaneous utility functions used across the pipeline.
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML config and return as dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ensure_dirs(config: dict) -> None:
    """Create all output directories defined in config if they don't exist."""
    dirs = [
        config["data"]["raw_dir"],
        config["data"]["processed_dir"],
        config["data"]["external_dir"],
        config["output"]["models_dir"],
        config["output"]["forecasts_dir"],
        config["output"]["reports_dir"],
        config["output"]["dashboards_dir"],
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print(f"[helpers] Ensured {len(dirs)} directories exist.")


def top_store_dept_pairs(
    df: pd.DataFrame,
    n: int = 10,
    target: str = "Weekly_Sales",
) -> list:
    """Return the top-n Store-Dept pairs by total sales volume."""
    ranked = (
        df.groupby(["Store", "Dept"])[target]
        .sum()
        .sort_values(ascending=False)
        .head(n)
    )
    return list(ranked.index)


def add_forecast_to_df(
    df: pd.DataFrame,
    store: int,
    dept: int,
    forecast_values: np.ndarray,
    start_date: pd.Timestamp,
    freq: str = "W",
) -> pd.DataFrame:
    """
    Append forecast rows to the main dataframe for a given Store-Dept.

    Parameters
    ----------
    df             : Existing processed dataframe.
    store          : Store number.
    dept           : Dept number.
    forecast_values: Array of predicted sales values.
    start_date     : Date of the first forecasted period.
    freq           : Pandas date frequency string.

    Returns
    -------
    Updated dataframe with forecast rows appended (is_forecast=True column).
    """
    dates = pd.date_range(start=start_date, periods=len(forecast_values), freq=freq)
    forecast_df = pd.DataFrame({
        "Store":        store,
        "Dept":         dept,
        "Date":         dates,
        "Weekly_Sales": forecast_values,
        "IsHoliday":    False,
        "is_forecast":  True,
    })
    df["is_forecast"] = df.get("is_forecast", False)
    return pd.concat([df, forecast_df], ignore_index=True)


def summarise_forecast_accuracy(results_list: list) -> pd.DataFrame:
    """
    Convert a list of model result dicts to a sorted summary DataFrame.

    Parameters
    ----------
    results_list : Output of evaluate_model() calls.

    Returns
    -------
    pd.DataFrame sorted by MAPE ascending.
    """
    df = pd.DataFrame(results_list).set_index("model")
    return df.sort_values("mape")
