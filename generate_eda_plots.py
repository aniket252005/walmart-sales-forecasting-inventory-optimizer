"""
generate_eda_plots.py
---------------------
Generates and saves all EDA visualizations to outputs/eda_plots/
Run: python generate_eda_plots.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from data.data_loader import load_config, load_processed_data

# ── Setup ────────────────────────────────────────────────────────────────────
cfg     = load_config()
df      = load_processed_data(cfg)
df      = df.reset_index()   # move Date from index back to column
OUT_DIR = Path("outputs/eda_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                     "savefig.facecolor": "white"})

df["Year"]  = pd.to_datetime(df["Date"]).dt.year
df["Month"] = pd.to_datetime(df["Date"]).dt.month

print(f"Data loaded: {df.shape[0]:,} rows | {df['Store'].nunique()} stores | {df['Dept'].nunique()} depts")
print(f"Saving plots to: {OUT_DIR.resolve()}\n")

# ── 1. Overall Sales Trend ───────────────────────────────────────────────────
print("1. Sales trend...")
weekly = df.groupby("Date")["Weekly_Sales"].sum().reset_index()
weekly["Date"] = pd.to_datetime(weekly["Date"])

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(weekly["Date"], weekly["Weekly_Sales"] / 1e6,
        linewidth=1.8, color="#1f77b4")
ax.fill_between(weekly["Date"], weekly["Weekly_Sales"] / 1e6,
                alpha=0.12, color="#1f77b4")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
ax.set_title("Total Weekly Sales Across All 45 Stores", fontsize=15, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Weekly Sales ($ Millions)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.1f}M"))
plt.tight_layout()
plt.savefig(OUT_DIR / "01_sales_trend.png")
plt.close()

# ── 2. Monthly Seasonality Heatmap ──────────────────────────────────────────
print("2. Monthly heatmap...")
pivot = df.pivot_table(values="Weekly_Sales", index="Year",
                       columns="Month", aggfunc="sum") / 1e6
pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                 "Jul","Aug","Sep","Oct","Nov","Dec"]
fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
            ax=ax, linewidths=0.5, cbar_kws={"label": "Sales ($M)"})
ax.set_title("Monthly Sales Heatmap — Total Sales per Month ($M)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "02_monthly_heatmap.png")
plt.close()

# ── 3. Sales Distribution ────────────────────────────────────────────────────
print("3. Sales distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["Weekly_Sales"], bins=80, color="#1f77b4",
             edgecolor="white", linewidth=0.3)
axes[0].set_title("Distribution of Weekly Sales", fontweight="bold")
axes[0].set_xlabel("Weekly Sales ($)"); axes[0].set_ylabel("Frequency")
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

month_data = [df[df["Month"] == m]["Weekly_Sales"].values for m in range(1, 13)]
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
axes[1].boxplot(month_data, labels=month_names, patch_artist=True,
                boxprops=dict(facecolor="#aec7e8"),
                medianprops=dict(color="#d62728", linewidth=2))
axes[1].set_title("Weekly Sales Distribution by Month", fontweight="bold")
axes[1].set_xlabel("Month"); axes[1].set_ylabel("Weekly Sales ($)")
axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(OUT_DIR / "03_sales_distribution.png")
plt.close()

# ── 4. Top 10 Stores ─────────────────────────────────────────────────────────
print("4. Top stores...")
store_totals = (df.groupby("Store")["Weekly_Sales"].sum()
                  .sort_values(ascending=False).head(10) / 1e6)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(store_totals.index.astype(str), store_totals.values,
              color=sns.color_palette("Blues_r", 10))
ax.set_title("Top 10 Stores by Total Sales (2010–2012)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Store Number"); ax.set_ylabel("Total Sales ($M)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}M"))
for bar, val in zip(bars, store_totals.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"${val:.1f}M", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "04_top_stores.png")
plt.close()

# ── 5. Holiday Effect ────────────────────────────────────────────────────────
print("5. Holiday effect...")
holiday_avg = df.groupby("IsHoliday")["Weekly_Sales"].mean().reset_index()
holiday_avg["Label"] = holiday_avg["IsHoliday"].map(
    {True: "Holiday Week", False: "Regular Week"})
fig, ax = plt.subplots(figsize=(7, 5))
colors = ["#4c72b0", "#dd8452"]
bars = ax.bar(holiday_avg["Label"], holiday_avg["Weekly_Sales"],
              color=colors, width=0.5, edgecolor="white")
ax.set_title("Avg Weekly Sales: Holiday vs Regular Weeks",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Avg Weekly Sales ($)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
for bar, val in zip(bars, holiday_avg["Weekly_Sales"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f"${val:,.0f}", ha="center", va="bottom", fontweight="bold")
pct = ((holiday_avg.iloc[1]["Weekly_Sales"] / holiday_avg.iloc[0]["Weekly_Sales"]) - 1) * 100
ax.set_xlabel(f"Holiday weeks are {pct:.1f}% higher on average", fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / "05_holiday_effect.png")
plt.close()

# ── 6. Yearly Sales Comparison ───────────────────────────────────────────────
print("6. Yearly comparison...")
yearly = df.groupby("Year")["Weekly_Sales"].agg(["sum","mean"]).reset_index()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(yearly["Year"].astype(str), yearly["sum"] / 1e9,
            color=["#1f77b4","#ff7f0e","#2ca02c"], edgecolor="white")
axes[0].set_title("Total Annual Sales", fontweight="bold")
axes[0].set_ylabel("Total Sales ($B)")
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.2f}B"))

axes[1].bar(yearly["Year"].astype(str), yearly["mean"],
            color=["#1f77b4","#ff7f0e","#2ca02c"], edgecolor="white")
axes[1].set_title("Avg Weekly Sales per Store-Dept", fontweight="bold")
axes[1].set_ylabel("Avg Weekly Sales ($)")
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.suptitle("Year-over-Year Sales Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "06_yearly_comparison.png")
plt.close()

# ── 7. Dept Sales Distribution ───────────────────────────────────────────────
print("7. Department analysis...")
dept_totals = (df.groupby("Dept")["Weekly_Sales"].sum()
                 .sort_values(ascending=False).head(15) / 1e6)
fig, ax = plt.subplots(figsize=(14, 5))
ax.barh(dept_totals.index.astype(str)[::-1], dept_totals.values[::-1],
        color=sns.color_palette("viridis", 15))
ax.set_title("Top 15 Departments by Total Sales", fontsize=14, fontweight="bold")
ax.set_xlabel("Total Sales ($M)")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}M"))
plt.tight_layout()
plt.savefig(OUT_DIR / "07_top_departments.png")
plt.close()

# ── 8. Time-Series Decomposition ─────────────────────────────────────────────
print("8. Decomposition (Store 1, Dept 1)...")
series = (df[(df["Store"] == 1) & (df["Dept"] == 1)]
          .set_index("Date")["Weekly_Sales"]
          .sort_index()
          .ffill())
series.index = pd.to_datetime(series.index)

decomp = seasonal_decompose(series, model="additive", period=52)
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
axes[0].plot(series.index, decomp.observed,  color="#1f77b4", lw=1.5)
axes[0].set_ylabel("Observed");   axes[0].set_title("Time-Series Decomposition — Store 1, Dept 1", fontweight="bold")
axes[1].plot(series.index, decomp.trend,     color="#ff7f0e", lw=1.5)
axes[1].set_ylabel("Trend")
axes[2].plot(series.index, decomp.seasonal,  color="#2ca02c", lw=1.5)
axes[2].set_ylabel("Seasonal")
axes[3].plot(series.index, decomp.resid,     color="#9467bd", lw=1.0, alpha=0.8)
axes[3].set_ylabel("Residual")
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.tight_layout()
plt.savefig(OUT_DIR / "08_decomposition.png")
plt.close()

# ── 9. Stationarity Tests ────────────────────────────────────────────────────
print("9. Stationarity (ADF test)...")
series_diff = series.diff().dropna()
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(series.index, series.values, color="#1f77b4", lw=1.5)
axes[0].set_title("Original Series (Store 1, Dept 1)", fontweight="bold")
axes[0].set_ylabel("Weekly Sales ($)")
adf_orig = adfuller(series.dropna())
axes[0].set_xlabel(f"ADF p-value: {adf_orig[1]:.4f} — "
                   f"{'Stationary ✓' if adf_orig[1] < 0.05 else 'Non-stationary ✗'}")

axes[1].plot(series_diff.index, series_diff.values, color="#d62728", lw=1.2, alpha=0.8)
axes[1].set_title("First-Differenced Series", fontweight="bold")
axes[1].set_ylabel("Δ Weekly Sales ($)")
adf_diff = adfuller(series_diff.dropna())
axes[1].set_xlabel(f"ADF p-value: {adf_diff[1]:.4f} — "
                   f"{'Stationary ✓' if adf_diff[1] < 0.05 else 'Non-stationary ✗'}")
plt.tight_layout()
plt.savefig(OUT_DIR / "09_stationarity.png")
plt.close()

# ── 10. Rolling Statistics ───────────────────────────────────────────────────
print("10. Rolling statistics...")
rolling_mean = series.rolling(window=13).mean()
rolling_std  = series.rolling(window=13).std()
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(series.index, series.values,       color="#1f77b4", lw=1.2, alpha=0.6, label="Actual")
ax.plot(rolling_mean.index, rolling_mean,  color="#ff7f0e", lw=2.0, label="13-Week Rolling Mean")
ax.fill_between(rolling_mean.index,
                rolling_mean - rolling_std,
                rolling_mean + rolling_std,
                alpha=0.15, color="#ff7f0e", label="±1 Std Dev")
ax.set_title("Rolling Mean & Std Dev (Store 1, Dept 1 — 13-Week Window)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Weekly Sales ($)")
ax.legend(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUT_DIR / "10_rolling_statistics.png")
plt.close()

print(f"\n✅ All 10 EDA plots saved to: {OUT_DIR.resolve()}")
files = list(OUT_DIR.glob("*.png"))
for f in sorted(files):
    print(f"   {f.name}")
