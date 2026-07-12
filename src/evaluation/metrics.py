"""
metrics.py
----------
Evaluation metrics for time-series forecasting models.
All functions accept array-like actual / predicted values.
"""

import numpy as np
import pandas as pd
from typing import Union

ArrayLike = Union[np.ndarray, pd.Series, list]


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Root Mean Squared Error — penalises large errors heavily."""
    a, p = np.array(actual), np.array(predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mae(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Mean Absolute Error — easy to interpret, robust to outliers."""
    a, p = np.array(actual), np.array(predicted)
    return float(np.mean(np.abs(a - p)))


def mape(actual: ArrayLike, predicted: ArrayLike, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error — scale-independent.
    epsilon avoids division by zero when actual == 0.

    Returns a percentage value (e.g., 12.3 means 12.3%).
    """
    a, p = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    return float(np.mean(np.abs((a - p) / (np.abs(a) + epsilon))) * 100)


def smape(actual: ArrayLike, predicted: ArrayLike, epsilon: float = 1e-8) -> float:
    """
    Symmetric MAPE — less biased than MAPE for under-forecasts.
    Returns a percentage value.
    """
    a, p = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    denominator = (np.abs(a) + np.abs(p)) / 2 + epsilon
    return float(np.mean(np.abs(a - p) / denominator) * 100)


def r2_score(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Coefficient of determination R². 1.0 = perfect, 0 = mean baseline."""
    a, p = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def naive_forecast(series: pd.Series) -> pd.Series:
    """
    Naive baseline: last observed value carried forward (persistence model).
    Returns a series of the same length, shifted by 1.
    """
    return series.shift(1)


def seasonal_naive_forecast(series: pd.Series, period: int = 52) -> pd.Series:
    """Seasonal naive: use value from the same period last cycle."""
    return series.shift(period)


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

def evaluate_model(
    actual: ArrayLike,
    predicted: ArrayLike,
    model_name: str = "Model",
) -> dict:
    """
    Compute all metrics and return as a dictionary.

    Parameters
    ----------
    actual      : Ground truth values.
    predicted   : Model predictions.
    model_name  : Label for the model (used in printed output).

    Returns
    -------
    dict with keys: model, rmse, mae, mape, smape, r2
    """
    results = {
        "model": model_name,
        "rmse":  round(rmse(actual, predicted), 4),
        "mae":   round(mae(actual, predicted), 4),
        "mape":  round(mape(actual, predicted), 2),
        "smape": round(smape(actual, predicted), 2),
        "r2":    round(r2_score(actual, predicted), 4),
    }
    print(
        f"[{model_name}] RMSE={results['rmse']:,.2f} | "
        f"MAE={results['mae']:,.2f} | "
        f"MAPE={results['mape']:.2f}% | "
        f"SMAPE={results['smape']:.2f}% | "
        f"R²={results['r2']:.4f}"
    )
    return results


def compare_models(results_list: list) -> pd.DataFrame:
    """
    Pretty-print a side-by-side comparison table.

    Parameters
    ----------
    results_list : list of dicts returned by evaluate_model()

    Returns
    -------
    pd.DataFrame sorted by MAPE ascending.
    """
    df = pd.DataFrame(results_list).set_index("model")
    df = df.sort_values("mape")
    print("\n=== Model Comparison ===")
    print(df.to_string())
    return df


if __name__ == "__main__":
    # Quick sanity check
    np.random.seed(42)
    actual    = np.random.uniform(100, 500, 50)
    predicted = actual + np.random.normal(0, 20, 50)

    res = evaluate_model(actual, predicted, "TestModel")
    print("\nAll metrics:", res)
