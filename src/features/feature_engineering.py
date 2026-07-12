"""
feature_engineering.py
-----------------------
Generates 25+ time-based, lag, rolling, and cyclic features from the
cleaned weekly sales data. Saves to data/processed/sales_features.csv.

Features created:
  Time-based : year, month, quarter, week_of_year, day_of_week, is_weekend
  Cyclic     : month_sin/cos, week_sin/cos (prevents distance issues at boundaries)
  Lag        : sales_lag_4w, sales_lag_8w, sales_lag_13w (configurable)
  Rolling    : rolling_mean_4w, rolling_std_4w, rolling_min_4w, rolling_max_4w
  Holiday    : is_holiday (from dataset), weeks_to_holiday, weeks_since_holiday
  Trend      : expanding_mean (running average up to t-1)
"""

import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

# Allow running as a script from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.data_loader import load_config, load_processed_data


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Extract calendar-based features."""
    d = pd.to_datetime(df[date_col])
    df["year"]         = d.dt.year
    df["month"]        = d.dt.month
    df["quarter"]      = d.dt.quarter
    df["week_of_year"] = d.dt.isocalendar().week.astype(int)
    df["day_of_week"]  = d.dt.dayofweek        # 0=Mon … 6=Sun
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    return df


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode periodic features as sin/cos pairs so that, e.g.,
    month=12 and month=1 are treated as adjacent.
    """
    df["month_sin"]    = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]    = np.cos(2 * np.pi * df["month"] / 12)
    df["week_sin"]     = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"]     = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["dow_sin"]      = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]      = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_lag_features(
    df: pd.DataFrame,
    lag_periods: list,
    group_cols: list = ["Store", "Dept"],
    target: str = "Weekly_Sales",
) -> pd.DataFrame:
    """
    Add lag features per Store-Dept group.
    lag_periods are in weeks (e.g. [4, 8, 13]).
    """
    for lag in lag_periods:
        col = f"sales_lag_{lag}w"
        df[col] = (
            df.groupby(group_cols)[target]
            .transform(lambda s: s.shift(lag))
        )
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: list,
    group_cols: list = ["Store", "Dept"],
    target: str = "Weekly_Sales",
) -> pd.DataFrame:
    """
    Add rolling mean, std, min, max for given window sizes (in weeks).
    Uses shift(1) so no current-week data leaks into the feature.
    """
    for w in windows:
        shifted = df.groupby(group_cols)[target].transform(lambda s: s.shift(1))
        df[f"rolling_mean_{w}w"] = (
            shifted.groupby(df[group_cols].apply(tuple, axis=1))
            .transform(lambda s: s.rolling(w, min_periods=1).mean())
        )
        df[f"rolling_std_{w}w"] = (
            shifted.groupby(df[group_cols].apply(tuple, axis=1))
            .transform(lambda s: s.rolling(w, min_periods=1).std().fillna(0))
        )
        df[f"rolling_min_{w}w"] = (
            shifted.groupby(df[group_cols].apply(tuple, axis=1))
            .transform(lambda s: s.rolling(w, min_periods=1).min())
        )
        df[f"rolling_max_{w}w"] = (
            shifted.groupby(df[group_cols].apply(tuple, axis=1))
            .transform(lambda s: s.rolling(w, min_periods=1).max())
        )
    return df


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive holiday proximity features from the IsHoliday flag.
    weeks_to_holiday   : weeks until next holiday week (0 if current is holiday)
    weeks_since_holiday: weeks since last holiday week
    """
    if "IsHoliday" not in df.columns:
        df["is_holiday"] = 0
        return df

    df["is_holiday"] = df["IsHoliday"].fillna(False).astype(int)

    # Compute within each Store-Dept group
    def _holiday_dist(group):
        holiday_idx = group.index[group["IsHoliday"] == 1].tolist()
        positions = np.arange(len(group))

        if not holiday_idx:
            group["weeks_to_holiday"]    = 99
            group["weeks_since_holiday"] = 99
            return group

        h_pos = [group.index.get_loc(i) for i in holiday_idx]

        to_next    = []
        since_last = []
        for pos in positions:
            future = [p - pos for p in h_pos if p >= pos]
            past   = [pos - p for p in h_pos if p <= pos]
            to_next.append(min(future) if future else 99)
            since_last.append(min(past) if past else 99)

        group["weeks_to_holiday"]    = to_next
        group["weeks_since_holiday"] = since_last
        return group

    df = df.groupby(["Store", "Dept"], group_keys=False).apply(_holiday_dist)
    return df


def add_expanding_mean(
    df: pd.DataFrame,
    group_cols: list = ["Store", "Dept"],
    target: str = "Weekly_Sales",
) -> pd.DataFrame:
    """Historical running mean up to but not including the current week."""
    df["expanding_mean"] = (
        df.groupby(group_cols)[target]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-feature interactions for tree-based models."""
    df["month_x_dow"]    = df["month"] * df["day_of_week"]
    df["quarter_x_week"] = df["quarter"] * df["week_of_year"]
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    lag_periods     = config["features"]["lag_periods"]
    rolling_windows = config["features"]["rolling_windows"]

    # Reset index if Date is the index
    if df.index.name == "Date":
        df = df.reset_index()

    df = add_time_features(df)
    df = add_cyclic_features(df)
    df = add_lag_features(df, lag_periods=lag_periods)
    df = add_rolling_features(df, windows=rolling_windows)
    df = add_holiday_features(df)
    df = add_expanding_mean(df)
    df = add_interaction_features(df)

    # Drop rows where lag/rolling features are NaN (first few weeks per group)
    lag_cols = [f"sales_lag_{p}w" for p in lag_periods]
    df = df.dropna(subset=lag_cols)
    df = df.reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"]]
    print(f"[feature_engineering] Total features created: {len(feature_cols)}")
    print(f"[feature_engineering] Features: {feature_cols}")
    print(f"[feature_engineering] Dataset shape after feature engineering: {df.shape}")
    return df


def save_features(df: pd.DataFrame, config: dict) -> None:
    out_dir  = Path(config["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / config["data"]["features_filename"]
    df.to_csv(out_path, index=False)
    print(f"[feature_engineering] Saved features → {out_path}")


if __name__ == "__main__":
    cfg        = load_config()
    processed  = load_processed_data(cfg)
    features   = engineer_features(processed, cfg)
    save_features(features, cfg)
    print("\nSample of feature-engineered data:")
    print(features.head())
