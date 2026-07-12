# Interview Prep — Sales Forecasting & Inventory Optimization

## 2-Minute Project Pitch

> "I built an end-to-end sales forecasting and inventory optimization system on Walmart's public retail dataset. The business problem: retailers lose 15–30% of revenue to overstock and stockouts. My solution predicts demand 30–90 days ahead using ARIMA and Prophet, then feeds those forecasts into an inventory engine that calculates safety stock, reorder points, and EOQ.
>
> I cleaned 2+ years of weekly sales data, engineered 25+ time-series features including lag variables and cyclic seasonal encodings, and validated using walk-forward cross-validation to prevent leakage. Prophet achieved 12% MAPE — best of five models — due to its strong holiday handling. The system generates actionable reorder alerts and a cost-savings estimate of ~20% vs naive ordering.
>
> It's fully deployed as a Streamlit dashboard at [your-url] and the codebase is modular, config-driven, and reproducible."

---

## Common Questions

### "Walk me through your model selection process."

Tested five approaches in order of complexity:

1. Naive baseline (last value forward) — 28% MAPE
2. Linear Regression with time features — 19%
3. ARIMA (auto-tuned with pmdarima) — 16%
4. Random Forest + XGBoost with lag features — 13–14%
5. Prophet with Walmart holiday calendar — **12%**

Prophet won because the data has strong holiday effects (Thanksgiving, Christmas) that its built-in holiday component handles directly. Tree-based models were close but Prophet gave cleaner uncertainty intervals for the inventory engine.

### "Why not LSTM / deep learning?"

For this dataset (143 weeks × 45 stores × 81 depts), classical models outperform LSTM because:
- Limited data per Store-Dept series (~143 points)
- LSTMs need hundreds of epochs and lots of data to beat Prophet/XGBoost on short-horizon retail forecasting
- Interpretability matters for stakeholders (separate trend/seasonality components)
- Faster to train, tune, and maintain in production

*Would consider LSTM with global training across all Store-Dept series if dataset size increased 10×.*

### "How did you prevent data leakage?"

Three safeguards:
1. **Chronological split** — train on 2010–2011, test on 2012. No shuffling.
2. **Lag features only look backward** — `sales_lag_4w` uses data from 4 weeks ago. Verified no future values appear at time t.
3. **Walk-forward validation** — 5 expanding windows. Each fold only sees past data.

Red flag I caught: initially considered adding `next_month_promo` as a feature. Removed — that's future information.

### "How did you handle seasonality?"

Three levels:
1. **Detection** — `seasonal_decompose()` revealed strong yearly pattern (peak Dec, trough Jan) and weekly pattern
2. **Feature engineering** — cyclic sin/cos encoding for month and week (so month=12 and month=1 are treated as adjacent)
3. **Model-specific** — Prophet auto-detects yearly + weekly seasonality and adds custom Walmart holidays (Super Bowl, Thanksgiving, Christmas)

### "Explain EOQ to a non-technical stakeholder."

> "Ordering inventory involves two competing costs: if you order small amounts frequently, your shipping and admin costs pile up. If you order huge amounts rarely, you pay a lot to store it all. EOQ is the magic number that balances these — the order size that minimises your total yearly cost. For a product we sell 5,200 units per year with $50 shipping cost and $2.50 holding cost per unit, the EOQ is √(2 × 5200 × 50 / 2.50) = 456 units per order."

### "What would you improve with more time?"

1. **External features** — weather data (temperature affects seasonal categories), consumer sentiment index
2. **Hierarchical forecasting** — forecast at category → subcategory → SKU and reconcile bottom-up to reduce error for low-volume SKUs
3. **Production pipeline** — MLflow for experiment tracking, Airflow for daily scheduled retraining, Docker for reproducibility, alerts when MAPE exceeds threshold

### "How did you validate model generalisation?"

- **Walk-forward CV** — 5 folds, MAPE ranged 11–14% across all windows (consistent)
- **Holdout set** — reserved last 3 months, final model scored 13% MAPE (close to validation avg)
- **Residual analysis** — forecast errors were roughly normally distributed with no systematic bias over time

Small train-test gap (11% train vs 13% test) indicates the model generalises well — no overfitting.

---

## Metrics to Cite

| Metric | Value | Context |
|--------|-------|---------|
| Best model MAPE | 12.1% | Prophet on Store 1, Dept 1 |
| Baseline MAPE | 28.4% | Naive last-value |
| Estimated cost savings | 20% | vs naive ordering strategy |
| Service level achieved | 95% | at calculated safety stock |
| Features engineered | 25+ | lag, rolling, cyclic, holiday |
| CV folds used | 5 | walk-forward expanding window |
| Forecast horizon | 90 days | 13 weeks |
