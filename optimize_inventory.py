"""
optimize_inventory.py
---------------------
Full inventory optimization pipeline. Loads processed sales data,
applies saved forecasts, and outputs inventory recommendations.

Run
---
    python optimize_inventory.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "src")

from data.data_loader                 import load_config, load_processed_data
from optimization.inventory_optimizer import (
    optimize_inventory,
    generate_reorder_alerts,
    save_recommendations,
)


def load_or_simulate_predictions(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Try to load saved forecast CSVs from outputs/forecasts/.
    Fall back to a slightly-noised copy of actuals for demo purposes.
    """
    forecasts_dir = Path(config["output"]["forecasts_dir"])
    forecast_files = list(forecasts_dir.glob("forecast_store*.csv")) if forecasts_dir.exists() else []

    if forecast_files:
        frames = [pd.read_csv(f, parse_dates=["Date"]) for f in forecast_files]
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.rename(columns={"Forecast": "Predicted_Sales"})
        print(f"[optimize_inventory] Loaded {len(forecast_files)} forecast file(s).")
        return combined

    # Demo fallback — simulate predictions
    print("[optimize_inventory] No forecast files found. Using simulated predictions.")
    np.random.seed(42)
    df_preds = df[["Store", "Dept", "Date", "Weekly_Sales"]].copy()
    df_preds["Predicted_Sales"] = df["Weekly_Sales"] * (1 + np.random.normal(0, 0.08, len(df)))
    df_preds["Predicted_Sales"] = df_preds["Predicted_Sales"].clip(lower=0)
    return df_preds


def simulate_current_inventory(recommendations: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate a current_inventory table for demo purposes.
    In production, this would come from your warehouse management system.
    """
    np.random.seed(7)
    inv = recommendations[["Store", "Dept", "reorder_point"]].copy()
    # Set 30% of products to be below their reorder point
    inv["Current_Stock"] = inv["reorder_point"] * np.random.uniform(0.5, 1.5, len(inv))
    return inv[["Store", "Dept", "Current_Stock"]]


if __name__ == "__main__":
    config = load_config()

    print("=" * 60)
    print("  Inventory Optimization Pipeline")
    print("=" * 60)

    # 1. Load processed data
    df = load_processed_data(config)

    # 2. Load or simulate predictions
    df_preds = load_or_simulate_predictions(df, config)

    # 3. Run optimization
    recommendations = optimize_inventory(df, df_preds, config)

    # 4. Summary statistics
    print("\n=== Inventory Optimization Summary ===")
    print(f"  Products optimized   : {len(recommendations)}")
    print(f"  Avg Safety Stock     : {recommendations['safety_stock'].mean():.1f} units")
    print(f"  Avg Reorder Point    : {recommendations['reorder_point'].mean():.1f} units")
    print(f"  Avg EOQ              : {recommendations['eoq'].mean():.1f} units")
    print(f"  Avg Cost Savings     : {recommendations['estimated_savings_pct'].mean():.1f}%")

    # 5. Generate reorder alerts
    current_inv = simulate_current_inventory(recommendations)
    alerts = generate_reorder_alerts(recommendations, current_inv)

    if not alerts.empty:
        print(f"\n=== Reorder Alerts: {len(alerts)} products below ROP ===")
        print(alerts[["Store", "Dept", "reorder_point", "Current_Stock", "eoq", "units_short"]]
              .head(20).to_string(index=False))

    # 6. Save
    save_recommendations(recommendations, config)

    alerts_dir = Path(config["output"]["forecasts_dir"])
    alerts_dir.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(alerts_dir / "reorder_alerts.csv", index=False)
    print(f"[optimize_inventory] Alerts saved → {alerts_dir / 'reorder_alerts.csv'}")

    print("\nDone. Check outputs/forecasts/ for results.")
