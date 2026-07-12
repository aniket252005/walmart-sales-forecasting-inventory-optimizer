"""
backtesting.py
--------------
Walk-forward (expanding window) cross-validation for time-series models.
Prevents data leakage by ensuring the model only trains on past data
at every validation step.

Usage
-----
    from backtesting import walk_forward_validation
    results = walk_forward_validation(model_fn, series, n_splits=5)
"""

import numpy as np
import pandas as pd
from typing import Callable, List, Dict
from sklearn.model_selection import TimeSeriesSplit

try:
    from metrics import evaluate_model, compare_models
except ModuleNotFoundError:
    from src.evaluation.metrics import evaluate_model, compare_models


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward_validation(
    train_fn: Callable,
    predict_fn: Callable,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    model_name: str = "Model",
) -> pd.DataFrame:
    """
    Expanding-window walk-forward cross-validation.

    Parameters
    ----------
    train_fn    : Callable(X_train, y_train) → fitted model
    predict_fn  : Callable(model, X_test) → np.ndarray of predictions
    X           : Feature matrix (must be sorted chronologically)
    y           : Target series (same index as X)
    n_splits    : Number of CV folds
    model_name  : Label for reporting

    Returns
    -------
    pd.DataFrame with per-fold metrics (rmse, mae, mape, smape, r2).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model      = train_fn(X_train, y_train)
        predictions = predict_fn(model, X_test)

        fold_metrics = evaluate_model(y_test, predictions, model_name=f"{model_name} Fold {fold}")
        fold_metrics["fold"]       = fold
        fold_metrics["train_size"] = len(train_idx)
        fold_metrics["test_size"]  = len(test_idx)
        fold_results.append(fold_metrics)

    summary = pd.DataFrame(fold_results).set_index("fold")

    # Aggregate
    numeric_cols = ["rmse", "mae", "mape", "smape", "r2"]
    print(f"\n=== Walk-Forward Summary: {model_name} ===")
    print(summary[numeric_cols].to_string())

    agg = summary[numeric_cols].agg(["mean", "std"])
    print(f"\nMean ± Std across {n_splits} folds:")
    for col in numeric_cols:
        print(f"  {col.upper():6s}: {agg.loc['mean', col]:.4f} ± {agg.loc['std', col]:.4f}")

    return summary


def simple_time_split(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list,
    split_ratio: float = 0.8,
) -> tuple:
    """
    Single chronological train/test split — no shuffling.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    split_idx = int(len(df) * split_ratio)
    train     = df.iloc[:split_idx]
    test      = df.iloc[split_idx:]

    X_train, y_train = train[feature_cols], train[target_col]
    X_test,  y_test  = test[feature_cols],  test[target_col]

    print(
        f"[backtesting] Train: {len(train):,} rows | "
        f"Test: {len(test):,} rows | "
        f"Split ratio: {split_ratio:.0%}"
    )
    return X_train, X_test, y_train, y_test


def get_feature_columns(df: pd.DataFrame, exclude: list = None) -> list:
    """Return all numeric columns except target, ID, and date columns."""
    default_exclude = ["Weekly_Sales", "Store", "Dept", "Date", "IsHoliday"]
    if exclude:
        default_exclude += exclude
    return [c for c in df.columns if c not in default_exclude and pd.api.types.is_numeric_dtype(df[c])]


if __name__ == "__main__":
    # Smoke test with a simple linear regression
    from sklearn.linear_model import LinearRegression
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.data_loader import load_config, load_features_data

    cfg = load_config()
    df  = load_features_data(cfg)

    # Use a single store-dept for demo
    sample = df[(df["Store"] == 1) & (df["Dept"] == 1)].copy()
    feature_cols = get_feature_columns(sample)
    sample = sample.dropna(subset=feature_cols)

    X = sample[feature_cols]
    y = sample["Weekly_Sales"]

    results = walk_forward_validation(
        train_fn    = lambda Xtr, ytr: LinearRegression().fit(Xtr, ytr),
        predict_fn  = lambda m, Xte: m.predict(Xte),
        X           = X,
        y           = y,
        n_splits    = 5,
        model_name  = "LinearRegression",
    )
