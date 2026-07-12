# System Architecture

## Data Flow

```
data/raw/train.csv
      │
      ▼
src/data/preprocessing.py      ← cleans, deduplicates, resamples
      │
      ▼
data/processed/sales_processed.csv
      │
      ▼
src/features/feature_engineering.py   ← 25+ lag, rolling, cyclic features
      │
      ▼
data/processed/sales_features.csv
      │
      ├──► src/models/regression_model.py  (Linear Regression, RF, XGBoost)
      ├──► src/models/arima_model.py        (auto_arima / SARIMA)
      └──► src/models/prophet_model.py      (Facebook Prophet)
                │
                ▼
         outputs/models/*.pkl
                │
                ▼
         src/evaluation/metrics.py + backtesting.py
                │
                ▼
         outputs/reports/model_comparison.csv
                │
                ▼
         src/optimization/inventory_optimizer.py
         (Safety Stock · ROP · EOQ)
                │
                ▼
         outputs/forecasts/
           ├── forecast_store*_dept*.csv
           ├── inventory_recommendations.csv
           └── reorder_alerts.csv
                │
                ▼
           app.py (Streamlit Dashboard)
```

## Module Descriptions

| Module | Purpose |
|--------|---------|
| `src/data/data_loader.py` | Reusable CSV loading, Store-Dept series extraction |
| `src/data/preprocessing.py` | Missing values, outlier removal, weekly resampling |
| `src/features/feature_engineering.py` | 25+ time, lag, rolling, cyclic, holiday features |
| `src/models/regression_model.py` | Linear Regression, Random Forest, XGBoost |
| `src/models/arima_model.py` | auto_arima with SARIMA seasonal support |
| `src/models/prophet_model.py` | Facebook Prophet with Walmart holiday calendar |
| `src/evaluation/metrics.py` | RMSE, MAE, MAPE, SMAPE, R² |
| `src/evaluation/backtesting.py` | Walk-forward cross-validation |
| `src/optimization/inventory_optimizer.py` | Safety Stock, ROP, EOQ, cost savings |
| `src/utils/logger.py` | Centralised logging to stdout + file |
| `src/utils/helpers.py` | Directory setup, series utilities |
| `main.py` | Full pipeline runner |
| `forecast.py` | Standalone forecast generator (CLI) |
| `optimize_inventory.py` | Standalone inventory optimizer (CLI) |
| `app.py` | Streamlit dashboard |

## Key Design Decisions

**No data leakage** — all splits are chronological. Lag features only look backward. Walk-forward validation simulates real deployment.

**Modular architecture** — each stage is an independent, importable module with a `if __name__ == "__main__"` demo block. You can run any stage in isolation.

**Config-driven** — all paths, hyperparameters, and business constants live in `config/config.yaml`. No hardcoded values in source files.

**Graceful fallbacks** — the Streamlit dashboard generates synthetic data if no processed CSV is found, so it always runs for demos.
