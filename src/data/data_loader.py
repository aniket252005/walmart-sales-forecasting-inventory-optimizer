"""
data_loader.py
--------------
Reusable module for loading raw and processed Walmart sales data.
Reads paths from config/config.yaml so nothing is hardcoded.
"""

import os
import yaml
import pandas as pd
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(config: dict) -> pd.DataFrame:
    """
    Load the raw Walmart sales CSV.

    Expected columns (Walmart Store Sales dataset):
        Store, Dept, Date, Weekly_Sales, IsHoliday

    Returns
    -------
    pd.DataFrame
        Raw dataframe with basic dtype fixes applied.
    """
    raw_path = Path(config["data"]["raw_dir"]) / config["data"]["raw_filename"]

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at '{raw_path}'.\n"
            "Download from Kaggle: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data\n"
            "Place train.csv inside data/raw/"
        )

    df = pd.read_csv(raw_path)

    # Ensure Date is parsed immediately
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)

    # Boolean coercion for IsHoliday
    if "IsHoliday" in df.columns:
        df["IsHoliday"] = df["IsHoliday"].astype(bool)

    print(f"[data_loader] Loaded raw data: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"[data_loader] Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"[data_loader] Stores: {df['Store'].nunique()}  |  Depts: {df['Dept'].nunique()}")
    return df


def load_processed_data(config: dict) -> pd.DataFrame:
    """Load the cleaned, processed dataset saved by preprocessing.py."""
    processed_path = (
        Path(config["data"]["processed_dir"]) / config["data"]["processed_filename"]
    )

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at '{processed_path}'.\n"
            "Run: python src/data/preprocessing.py"
        )

    df = pd.read_csv(processed_path, parse_dates=["Date"], index_col="Date")
    print(f"[data_loader] Loaded processed data: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def load_features_data(config: dict) -> pd.DataFrame:
    """Load the feature-engineered dataset saved by feature_engineering.py."""
    features_path = (
        Path(config["data"]["processed_dir"]) / config["data"]["features_filename"]
    )

    if not features_path.exists():
        raise FileNotFoundError(
            f"Features data not found at '{features_path}'.\n"
            "Run: python src/features/feature_engineering.py"
        )

    df = pd.read_csv(features_path, parse_dates=["Date"], index_col="Date")
    print(f"[data_loader] Loaded features data: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def get_store_dept_series(
    df: pd.DataFrame,
    store: int,
    dept: int,
    target_col: str = "Weekly_Sales",
) -> pd.Series:
    """
    Extract a single Store-Dept time series from the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns [Store, Dept, Date, Weekly_Sales, ...]
    store : int
        Store number.
    dept : int
        Department number.
    target_col : str
        Column to return as a Series.

    Returns
    -------
    pd.Series indexed by Date, sorted chronologically.
    """
    mask = (df["Store"] == store) & (df["Dept"] == dept)
    series = df.loc[mask].set_index("Date")[target_col].sort_index()

    if series.empty:
        raise ValueError(f"No data found for Store={store}, Dept={dept}.")

    print(f"[data_loader] Series for Store={store} Dept={dept}: {len(series)} weeks")
    return series


if __name__ == "__main__":
    cfg = load_config()
    raw = load_raw_data(cfg)
    print("\nColumn dtypes:")
    print(raw.dtypes)
    print("\nFirst 5 rows:")
    print(raw.head())
    print("\nMissing values:")
    print(raw.isnull().sum())
    print("\nBasic statistics:")
    print(raw["Weekly_Sales"].describe())
