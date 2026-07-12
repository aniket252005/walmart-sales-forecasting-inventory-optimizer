"""
preprocessing.py
----------------
Cleans the raw Walmart sales dataset and saves the result to
data/processed/sales_processed.csv.

Steps:
  1. Parse dates, sort chronologically
  2. Remove duplicates
  3. Handle missing values (forward-fill, then back-fill)
  4. Remove outliers via IQR or z-score
  5. Aggregate to consistent weekly frequency per Store-Dept
  6. Save cleaned data
"""

import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

try:
    from data_loader import load_config, load_raw_data
except ModuleNotFoundError:
    from src.data.data_loader import load_config, load_raw_data


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["Store", "Dept", "Date"])
    removed = before - len(df)
    if removed:
        print(f"[preprocessing] Removed {removed} duplicate rows.")
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = "forward_fill") -> pd.DataFrame:
    """
    Fill missing Weekly_Sales values within each Store-Dept group.
    strategy: 'forward_fill' | 'mean' | 'median'
    """
    missing_before = df["Weekly_Sales"].isnull().sum()

    if strategy == "forward_fill":
        df["Weekly_Sales"] = (
            df.groupby(["Store", "Dept"])["Weekly_Sales"]
            .transform(lambda s: s.ffill().bfill())
        )
    elif strategy in ("mean", "median"):
        fill_fn = "mean" if strategy == "mean" else "median"
        df["Weekly_Sales"] = (
            df.groupby(["Store", "Dept"])["Weekly_Sales"]
            .transform(lambda s: s.fillna(getattr(s, fill_fn)()))
        )

    missing_after = df["Weekly_Sales"].isnull().sum()
    print(
        f"[preprocessing] Missing values: {missing_before} → {missing_after} "
        f"(strategy='{strategy}')"
    )
    return df


def remove_outliers_iqr(df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    """Cap outliers at [Q1 - k*IQR, Q3 + k*IQR] per Store-Dept group."""
    def _cap(series):
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        return series.clip(lower=lower, upper=upper)

    df["Weekly_Sales"] = (
        df.groupby(["Store", "Dept"])["Weekly_Sales"].transform(_cap)
    )
    print(f"[preprocessing] Outliers capped using IQR × {multiplier}.")
    return df


def remove_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Remove rows where |z-score| > threshold within each Store-Dept group."""
    def _mask(series):
        z = np.abs(stats.zscore(series.dropna()))
        return z < threshold

    before = len(df)
    keep = df.groupby(["Store", "Dept"])["Weekly_Sales"].transform(
        lambda s: np.abs(stats.zscore(s.fillna(s.mean()))) < threshold
    )
    df = df[keep.astype(bool)].copy()
    print(
        f"[preprocessing] Removed {before - len(df)} outlier rows "
        f"(z-score threshold={threshold})."
    )
    return df


def enforce_weekly_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample each Store-Dept series to weekly frequency (W-FRI matches Walmart).
    Sums are used for Weekly_Sales; IsHoliday is OR-aggregated.
    """
    records = []
    for (store, dept), group in df.groupby(["Store", "Dept"]):
        group = group.set_index("Date").sort_index()
        resampled = group["Weekly_Sales"].resample("W").sum()
        holiday = group["IsHoliday"].resample("W").max()
        tmp = pd.DataFrame({"Weekly_Sales": resampled, "IsHoliday": holiday})
        tmp["Store"] = store
        tmp["Dept"] = dept
        records.append(tmp)

    result = pd.concat(records).reset_index()
    result = result.rename(columns={"index": "Date"})
    print(f"[preprocessing] After resampling: {result.shape[0]:,} rows.")
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def preprocess(config: dict) -> pd.DataFrame:
    df = load_raw_data(config)

    # 1. Sort
    df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

    # 2. Deduplicate
    df = remove_duplicates(df)

    # 3. Handle missing values
    strategy = config["preprocessing"]["missing_value_strategy"]
    df = handle_missing_values(df, strategy=strategy)

    # 4. Remove outliers
    method = config["preprocessing"]["outlier_method"]
    if method == "iqr":
        df = remove_outliers_iqr(df, multiplier=config["preprocessing"]["iqr_multiplier"])
    elif method == "zscore":
        df = remove_outliers_zscore(df, threshold=config["preprocessing"]["zscore_threshold"])

    # 5. Ensure consistent weekly frequency
    df = enforce_weekly_frequency(df)

    # 6. Remove any remaining NaN rows
    df = df.dropna(subset=["Weekly_Sales"])

    # 7. Ensure non-negative sales
    df["Weekly_Sales"] = df["Weekly_Sales"].clip(lower=0)

    return df


def save_processed(df: pd.DataFrame, config: dict) -> None:
    out_dir = Path(config["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / config["data"]["processed_filename"]
    df.to_csv(out_path, index=False)
    print(f"[preprocessing] Saved processed data → {out_path}")


if __name__ == "__main__":
    cfg = load_config()
    cleaned = preprocess(cfg)

    print(f"\nFinal shape: {cleaned.shape}")
    print(f"Date range : {cleaned['Date'].min().date()} → {cleaned['Date'].max().date()}")
    print(f"Stores     : {cleaned['Store'].nunique()}")
    print(f"Depts      : {cleaned['Dept'].nunique()}")
    print(f"Missing    : {cleaned.isnull().sum().sum()}")
    print(cleaned.head())

    save_processed(cleaned, cfg)
