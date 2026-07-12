"""
inventory_optimizer.py
----------------------
Calculates inventory optimization parameters for each Store-Dept:

  - Safety Stock  : Z × σ × √Lead Time
  - Reorder Point : (Avg Daily Demand × Lead Time) + Safety Stock
  - EOQ           : √(2 × D × S / H)

Also generates reorder alerts and estimates cost savings vs naive ordering.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


# ---------------------------------------------------------------------------
# Core formulas
# ---------------------------------------------------------------------------

def compute_safety_stock(
    forecast_errors: np.ndarray,
    lead_time_days: int,
    service_level: float = 0.95,
) -> float:
    """
    Safety Stock = Z × σ_demand × √(Lead Time in weeks)

    Parameters
    ----------
    forecast_errors : Residuals between actuals and model predictions.
    lead_time_days  : Days from placing an order to receiving it.
    service_level   : Desired probability of not stocking out (0–1).

    Returns
    -------
    float — safety stock in the same units as Weekly_Sales.
    """
    z_score      = stats.norm.ppf(service_level)
    sigma_demand = np.std(forecast_errors, ddof=1)
    lead_time_wk = lead_time_days / 7.0          # convert to weeks
    return float(z_score * sigma_demand * np.sqrt(lead_time_wk))


def compute_reorder_point(
    avg_weekly_demand: float,
    lead_time_days: int,
    safety_stock: float,
) -> float:
    """
    ROP = (Avg Weekly Demand × Lead Time in weeks) + Safety Stock

    When inventory drops to ROP, place a new order.
    """
    lead_time_wk = lead_time_days / 7.0
    return float(avg_weekly_demand * lead_time_wk + safety_stock)


def compute_eoq(
    annual_demand: float,
    order_cost: float,
    holding_cost_per_unit: float,
) -> float:
    """
    EOQ = √(2 × D × S / H)

    Parameters
    ----------
    annual_demand         : Total units sold per year.
    order_cost            : Fixed cost to place one order ($).
    holding_cost_per_unit : Annual cost to store one unit ($/unit/year).

    Returns
    -------
    float — optimal order quantity per purchase.
    """
    if holding_cost_per_unit <= 0 or annual_demand <= 0:
        return 0.0
    return float(np.sqrt(2 * annual_demand * order_cost / holding_cost_per_unit))


def compute_total_inventory_cost(
    annual_demand: float,
    order_quantity: float,
    order_cost: float,
    holding_cost_per_unit: float,
) -> float:
    """
    Total Cost = (D/Q) × S + (Q/2) × H
    """
    if order_quantity <= 0:
        return float("inf")
    ordering_cost = (annual_demand / order_quantity) * order_cost
    holding_cost  = (order_quantity / 2) * holding_cost_per_unit
    return float(ordering_cost + holding_cost)


# ---------------------------------------------------------------------------
# Per-product optimizer
# ---------------------------------------------------------------------------

def optimize_single_product(
    store: int,
    dept: int,
    actuals: np.ndarray,
    predictions: np.ndarray,
    config: dict,
) -> dict:
    """
    Compute all inventory parameters for one Store-Dept.

    Returns
    -------
    dict with keys: store, dept, avg_weekly_demand, annual_demand,
                    safety_stock, reorder_point, eoq,
                    current_cost_naive, optimal_cost, estimated_savings_pct
    """
    inv_cfg = config["inventory"]

    forecast_errors   = np.array(actuals) - np.array(predictions)
    avg_weekly_demand = float(np.mean(actuals))
    annual_demand     = avg_weekly_demand * 52

    safety_stock = compute_safety_stock(
        forecast_errors,
        lead_time_days = inv_cfg["default_lead_time_days"],
        service_level  = config["features"]["service_level"],
    )

    rop = compute_reorder_point(
        avg_weekly_demand = avg_weekly_demand,
        lead_time_days    = inv_cfg["default_lead_time_days"],
        safety_stock      = safety_stock,
    )

    holding_cost_per_unit = inv_cfg["default_unit_cost"] * inv_cfg["holding_cost_rate"]
    eoq = compute_eoq(
        annual_demand         = annual_demand,
        order_cost            = inv_cfg["order_cost"],
        holding_cost_per_unit = holding_cost_per_unit,
    )

    # Naive: order same fixed qty every week (avg weekly demand)
    naive_qty   = avg_weekly_demand
    naive_cost  = compute_total_inventory_cost(annual_demand, naive_qty, inv_cfg["order_cost"], holding_cost_per_unit)
    optimal_cost = compute_total_inventory_cost(annual_demand, eoq or naive_qty, inv_cfg["order_cost"], holding_cost_per_unit)
    savings_pct  = ((naive_cost - optimal_cost) / (naive_cost + 1e-8)) * 100 if naive_cost > 0 else 0.0

    return {
        "store":              store,
        "dept":               dept,
        "avg_weekly_demand":  round(avg_weekly_demand, 2),
        "annual_demand":      round(annual_demand, 2),
        "safety_stock":       round(safety_stock, 2),
        "reorder_point":      round(rop, 2),
        "eoq":                round(eoq, 2),
        "naive_annual_cost":  round(naive_cost, 2),
        "optimal_annual_cost":round(optimal_cost, 2),
        "estimated_savings_pct": round(savings_pct, 2),
    }


# ---------------------------------------------------------------------------
# Batch optimizer
# ---------------------------------------------------------------------------

def optimize_inventory(
    df_actuals: pd.DataFrame,
    df_predictions: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Run inventory optimization for all Store-Dept combinations.

    Parameters
    ----------
    df_actuals    : DataFrame with [Store, Dept, Date, Weekly_Sales].
    df_predictions: DataFrame with [Store, Dept, Date, Predicted_Sales].
    config        : Loaded config dict.

    Returns
    -------
    pd.DataFrame with one row per Store-Dept containing all inventory params.
    """
    results = []
    groups  = df_actuals.groupby(["Store", "Dept"])

    for (store, dept), actual_group in groups:
        pred_mask = (df_predictions["Store"] == store) & (df_predictions["Dept"] == dept)
        pred_group = df_predictions[pred_mask]

        if pred_group.empty:
            continue

        # Align on Date
        merged = actual_group.merge(
            pred_group[["Date", "Predicted_Sales"]],
            on="Date",
            how="inner",
        )
        if len(merged) < 4:
            continue   # not enough data

        result = optimize_single_product(
            store       = store,
            dept        = dept,
            actuals     = merged["Weekly_Sales"].values,
            predictions = merged["Predicted_Sales"].values,
            config      = config,
        )
        results.append(result)

    recommendations = pd.DataFrame(results)
    print(
        f"[inventory_optimizer] Optimized {len(recommendations)} Store-Dept combinations."
    )
    return recommendations


def generate_reorder_alerts(
    recommendations: pd.DataFrame,
    current_inventory: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flag any Store-Dept where current inventory is at or below the reorder point.

    Parameters
    ----------
    recommendations  : Output of optimize_inventory().
    current_inventory: DataFrame with [Store, Dept, Current_Stock].

    Returns
    -------
    Filtered DataFrame of products that need reordering, sorted by urgency.
    """
    merged = recommendations.merge(current_inventory, on=["Store", "Dept"], how="left")
    merged["below_rop"]  = merged["Current_Stock"] <= merged["reorder_point"]
    merged["units_short"] = (merged["reorder_point"] - merged["Current_Stock"]).clip(lower=0)
    alerts = merged[merged["below_rop"]].sort_values("units_short", ascending=False)

    print(f"[inventory_optimizer] {len(alerts)} products need reordering.")
    return alerts


def save_recommendations(recommendations: pd.DataFrame, config: dict) -> None:
    out_dir = Path(config["output"]["forecasts_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "inventory_recommendations.csv"
    recommendations.to_csv(path, index=False)
    print(f"[inventory_optimizer] Saved recommendations → {path}")


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.data_loader import load_config, load_processed_data

    cfg = load_config()
    df  = load_processed_data(cfg)

    # Simulate predictions = actuals + small noise
    np.random.seed(42)
    df_preds = df.copy()
    df_preds["Predicted_Sales"] = df["Weekly_Sales"] * (1 + np.random.normal(0, 0.1, len(df)))

    recs = optimize_inventory(df, df_preds, cfg)
    print(recs.head(10).to_string())
    save_recommendations(recs, cfg)
