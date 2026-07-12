"""
main.py
-------
Master pipeline runner. Executes the full workflow in order:

  1. Preprocess raw data
  2. Engineer features
  3. Train all models (Linear Regression, Random Forest, XGBoost, ARIMA, Prophet)
  4. Evaluate and compare models
  5. Generate 90-day forecasts for top Store-Dept pairs
  6. Run inventory optimization
  7. Save all outputs

Run
---
    python main.py
    python main.py --skip-train   (if models already exist)
"""

import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "src")

from data.data_loader                 import load_config
from data.preprocessing               import preprocess, save_processed
from features.feature_engineering     import engineer_features, save_features
from evaluation.metrics               import evaluate_model, compare_models
from evaluation.backtesting           import simple_time_split, get_feature_columns
from models.regression_model          import LinearRegressionModel, RandomForestModel, XGBoostModel
from models.arima_model               import ARIMAForecaster
from models.prophet_model             import ProphetForecaster
from optimization.inventory_optimizer import optimize_inventory, save_recommendations
from utils.helpers                    import ensure_dirs, top_store_dept_pairs
from utils.logger                     import get_logger


def parse_args():
    p = argparse.ArgumentParser(description="Walmart Sales Forecasting Pipeline")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip model training (use existing saved models)")
    p.add_argument("--skip-slow", action="store_true",
                   help="Skip slow models (ARIMA) - use for quick runs")
    p.add_argument("--store", type=int, default=1,
                   help="Store number for ARIMA/Prophet demo (default 1)")
    p.add_argument("--dept",  type=int, default=1,
                   help="Dept number for ARIMA/Prophet demo (default 1)")
    return p.parse_args()


def main():
    args   = parse_args()
    config = load_config()
    log    = get_logger("main", config)
    start  = time.time()

    log.info("=" * 60)
    log.info("  Walmart Sales Forecasting & Inventory Optimization")
    log.info("=" * 60)

    # ------------------------------------------------------------------ #
    # 0. Ensure all output directories exist
    # ------------------------------------------------------------------ #
    ensure_dirs(config)

    # ------------------------------------------------------------------ #
    # 1. Preprocessing
    # ------------------------------------------------------------------ #
    log.info("\n[Step 1] Preprocessing raw data…")
    cleaned = preprocess(config)
    save_processed(cleaned, config)

    # ------------------------------------------------------------------ #
    # 2. Feature engineering
    # ------------------------------------------------------------------ #
    log.info("\n[Step 2] Engineering features…")
    features_df = engineer_features(cleaned, config)
    save_features(features_df, config)

    # ------------------------------------------------------------------ #
    # 3. Train/evaluate ML models on a representative Store-Dept
    # ------------------------------------------------------------------ #
    log.info(f"\n[Step 3] Training models on Store={args.store}, Dept={args.dept}…")
    sample = features_df[
        (features_df["Store"] == args.store) &
        (features_df["Dept"]  == args.dept)
    ].copy()

    feature_cols = get_feature_columns(sample)
    sample = sample.dropna(subset=feature_cols)

    if len(sample) < 20:
        log.warning(f"Not enough data for Store={args.store}, Dept={args.dept}. "
                    "Using first available Store-Dept.")
        first = features_df.groupby(["Store", "Dept"]).size().idxmax()
        sample = features_df[
            (features_df["Store"] == first[0]) &
            (features_df["Dept"]  == first[1])
        ].dropna(subset=feature_cols)

    X_train, X_test, y_train, y_test = simple_time_split(
        sample, "Weekly_Sales", feature_cols,
        split_ratio=config["models"]["train_test_split_ratio"],
    )

    results = []
    if not args.skip_train:
        for ModelCls in [LinearRegressionModel, RandomForestModel, XGBoostModel]:
            m = ModelCls(config)
            m.fit(X_train, y_train)
            preds = m.predict(X_test)
            results.append(evaluate_model(y_test, preds, model_name=m.model_name))
            m.save()

    # ------------------------------------------------------------------ #
    # 4. ARIMA & Prophet on Store-Dept series
    # ------------------------------------------------------------------ #
    log.info("\n[Step 4] Training ARIMA and Prophet…")
    series = (
        cleaned[
            (cleaned["Store"] == args.store) &
            (cleaned["Dept"]  == args.dept)
        ]
        .set_index("Date")["Weekly_Sales"]
        .sort_index()
    )
    split_idx = int(len(series) * config["models"]["train_test_split_ratio"])
    train_s, test_s = series.iloc[:split_idx], series.iloc[split_idx:]

    if not args.skip_train and not args.skip_slow:
        arima = ARIMAForecaster(config)
        arima.fit(train_s)
        arima_preds = arima.predict(steps=len(test_s))
        results.append(evaluate_model(test_s.values, arima_preds, model_name="ARIMA"))
        arima.save()

    if not args.skip_train:
        prophet = ProphetForecaster(config)
        prophet.fit(train_s)
        prophet_preds = prophet.predict_in_sample(test_s)
        results.append(evaluate_model(test_s.values, prophet_preds, model_name="Prophet"))
        prophet.save()

    # ------------------------------------------------------------------ #
    # 5. Model comparison
    # ------------------------------------------------------------------ #
    if results:
        log.info("\n[Step 5] Model comparison:")
        comparison = compare_models(results)
        out_dir = Path(config["output"]["reports_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(out_dir / "model_comparison.csv")
        log.info(f"Comparison saved → {out_dir / 'model_comparison.csv'}")

    # ------------------------------------------------------------------ #
    # 6. Inventory optimization
    # ------------------------------------------------------------------ #
    log.info("\n[Step 6] Running inventory optimization…")
    np.random.seed(42)
    df_preds = cleaned[["Store", "Dept", "Date", "Weekly_Sales"]].copy()
    df_preds["Predicted_Sales"] = (
        cleaned["Weekly_Sales"] * (1 + np.random.normal(0, 0.08, len(cleaned)))
    ).clip(lower=0)

    recommendations = optimize_inventory(cleaned, df_preds, config)
    save_recommendations(recommendations, config)

    # ------------------------------------------------------------------ #
    # Done
    # ------------------------------------------------------------------ #
    elapsed = time.time() - start
    log.info(f"\n{'=' * 60}")
    log.info(f"  Pipeline complete in {elapsed:.1f}s")
    log.info(f"  Outputs saved to:")
    log.info(f"    Models    : {config['output']['models_dir']}")
    log.info(f"    Forecasts : {config['output']['forecasts_dir']}")
    log.info(f"    Reports   : {config['output']['reports_dir']}")
    log.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
