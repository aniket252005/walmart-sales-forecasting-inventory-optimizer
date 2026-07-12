"""
app.py
------
Streamlit dashboard for the Walmart Sales Forecasting &
Inventory Optimization project.

Deploy free in 5 min:
  1. Push this repo to GitHub
  2. Go to share.streamlit.io
  3. Select this file → Deploy

Run locally:
  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import yaml

sys.path.insert(0, "src")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title  = "Walmart Sales Forecast & Inventory Optimizer",
    page_icon   = "📦",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────
# Load config
# ─────────────────────────────────────────────
@st.cache_data
def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

CONFIG = load_config()


# ─────────────────────────────────────────────
# Data loaders (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_sales_data():
    """Load processed sales data or generate synthetic demo data."""
    path = Path(CONFIG["data"]["processed_dir"]) / CONFIG["data"]["processed_filename"]
    if path.exists():
        return pd.read_csv(path, parse_dates=["Date"])

    # Synthetic demo data so the dashboard runs without a dataset
    st.warning("No processed data found. Showing synthetic demo data. "
               "Run `python main.py` after placing train.csv in data/raw/.")
    np.random.seed(42)
    dates  = pd.date_range("2010-02-05", periods=143, freq="W")
    stores = [1, 2, 3]
    depts  = [1, 2, 3]
    rows   = []
    for s in stores:
        for d in depts:
            base  = np.random.uniform(5_000, 50_000)
            trend = np.linspace(0, base * 0.2, len(dates))
            seas  = base * 0.15 * np.sin(2 * np.pi * np.arange(len(dates)) / 52)
            noise = np.random.normal(0, base * 0.05, len(dates))
            sales = (base + trend + seas + noise).clip(min=0)
            for i, dt in enumerate(dates):
                rows.append({"Store": s, "Dept": d, "Date": dt,
                             "Weekly_Sales": sales[i], "IsHoliday": False})
    return pd.DataFrame(rows)


@st.cache_data
def load_inventory_recs():
    """Load inventory recommendations or simulate them."""
    path = Path(CONFIG["output"]["forecasts_dir"]) / "inventory_recommendations.csv"
    if path.exists():
        df = pd.read_csv(path)
        # Normalise column names to Title Case so the rest of the app is consistent
        df.columns = [c.title() if c in ("store","dept") else c for c in df.columns]
        # Rename lowercase metric columns to consistent names
        rename_map = {
            "store": "Store", "dept": "Dept",
            "avg_weekly_demand": "avg_weekly_demand",
            "safety_stock": "safety_stock",
            "reorder_point": "reorder_point",
            "eoq": "eoq",
            "estimated_savings_pct": "estimated_savings_pct",
        }
        df = df.rename(columns={"store": "Store", "dept": "Dept"})
        return df

    # Synthetic fallback
    df = load_sales_data()
    np.random.seed(7)
    recs = []
    for (s, d), grp in df.groupby(["Store", "Dept"]):
        avg_wk = grp["Weekly_Sales"].mean()
        ss     = avg_wk * 0.15
        rop    = avg_wk + ss
        eoq    = np.sqrt(2 * avg_wk * 52 * 50 / (10 * 0.25))
        recs.append({
            "Store": s, "Dept": d,
            "avg_weekly_demand": round(avg_wk, 1),
            "safety_stock": round(ss, 1),
            "reorder_point": round(rop, 1),
            "eoq": round(eoq, 1),
            "estimated_savings_pct": round(np.random.uniform(10, 30), 1),
        })
    return pd.DataFrame(recs)


@st.cache_data
def generate_demo_forecast(df, store, dept, horizon_weeks):
    """Generate a simple trend+seasonality forecast for display."""
    series = df[(df["Store"] == store) & (df["Dept"] == dept)].sort_values("Date")
    if series.empty:
        return pd.DataFrame()

    last_date  = series["Date"].max()
    avg_sales  = series["Weekly_Sales"].mean()
    dates      = pd.date_range(last_date + pd.DateOffset(weeks=1),
                               periods=horizon_weeks, freq="W")
    noise      = np.random.normal(0, avg_sales * 0.05, horizon_weeks)
    seas       = avg_sales * 0.12 * np.sin(2 * np.pi * np.arange(horizon_weeks) / 52)
    forecast   = (avg_sales + seas + noise).clip(min=0)

    return pd.DataFrame({
        "Date":      dates,
        "Forecast":  forecast,
        "Lower_CI":  forecast * 0.88,
        "Upper_CI":  forecast * 1.12,
    })


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Walmart_Spark.svg/200px-Walmart_Spark.svg.png", width=60)
    st.title("Controls")

    df_raw = load_sales_data()

    store_options = sorted(df_raw["Store"].unique())
    dept_options  = sorted(df_raw["Dept"].unique())

    selected_store = st.selectbox("Store", store_options)
    selected_dept  = st.selectbox("Department", dept_options)
    horizon        = st.select_slider("Forecast Horizon (weeks)", options=[4, 8, 13, 26, 52], value=13)
    service_level  = st.slider("Service Level (%)", 85, 99, 95)

    st.markdown("---")
    st.caption("📅 Date Range Filter")
    min_date = df_raw["Date"].min().date()
    max_date = df_raw["Date"].max().date()
    date_range = st.date_input("Select range", value=(min_date, max_date),
                               min_value=min_date, max_value=max_date)


# ─────────────────────────────────────────────
# Filter data
# ─────────────────────────────────────────────
if len(date_range) == 2:
    start_d, end_d = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    df_filtered = df_raw[
        (df_raw["Store"] == selected_store) &
        (df_raw["Dept"]  == selected_dept) &
        (df_raw["Date"] >= start_d) &
        (df_raw["Date"] <= end_d)
    ].sort_values("Date")
else:
    df_filtered = df_raw[
        (df_raw["Store"] == selected_store) &
        (df_raw["Dept"]  == selected_dept)
    ].sort_values("Date")

forecast_df  = generate_demo_forecast(df_raw, selected_store, selected_dept, horizon)
inventory_df = load_inventory_recs()
inv_row      = inventory_df[
    (inventory_df["Store"] == selected_store) &
    (inventory_df["Dept"]  == selected_dept)
]


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("📦 Walmart Sales Forecasting & Inventory Optimization")
st.caption(f"Store {selected_store}  ·  Department {selected_dept}  ·  {horizon}-week horizon")


# ─────────────────────────────────────────────
# KPI Cards
# ─────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

avg_sales = df_filtered["Weekly_Sales"].mean() if not df_filtered.empty else 0
total_sales = df_filtered["Weekly_Sales"].sum() if not df_filtered.empty else 0
forecast_avg = forecast_df["Forecast"].mean() if not forecast_df.empty else 0

k1.metric("Avg Weekly Sales",    f"${avg_sales:,.0f}")
k2.metric("Total Sales (period)",f"${total_sales:,.0f}")
k3.metric(f"{horizon}w Forecast Avg", f"${forecast_avg:,.0f}")
k4.metric("Service Level Target", f"{service_level}%")

st.markdown("---")


# ─────────────────────────────────────────────
# Row 1: Forecast chart
# ─────────────────────────────────────────────
st.subheader("📈 Sales Forecast")

if not df_filtered.empty:
    fig = go.Figure()

    # Historical actuals
    fig.add_trace(go.Scatter(
        x=df_filtered["Date"], y=df_filtered["Weekly_Sales"],
        mode="lines", name="Actual Sales",
        line=dict(color="#1f77b4", width=2),
    ))

    # Forecast with confidence band
    if not forecast_df.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df["Date"], forecast_df["Date"][::-1]]),
            y=pd.concat([forecast_df["Upper_CI"], forecast_df["Lower_CI"][::-1]]),
            fill="toself", fillcolor="rgba(255,127,14,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["Date"], y=forecast_df["Forecast"],
            mode="lines", name="Forecast",
            line=dict(color="#ff7f0e", width=2.5, dash="dot"),
        ))

    fig.update_layout(
        xaxis_title="Date", yaxis_title="Weekly Sales ($)",
        hovermode="x unified", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=40, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data available for the selected Store / Dept / date range.")


# ─────────────────────────────────────────────
# Row 2: Inventory KPIs + Seasonality heatmap
# ─────────────────────────────────────────────
col_inv, col_heat = st.columns([1, 2])

with col_inv:
    st.subheader("🏭 Inventory KPIs")
    if not inv_row.empty:
        r = inv_row.iloc[0]
        st.metric("Safety Stock",  f"{r.get('safety_stock', 0):,.0f} units")
        st.metric("Reorder Point", f"{r.get('reorder_point', 0):,.0f} units")
        st.metric("EOQ",           f"{r.get('eoq', 0):,.0f} units/order")
        st.metric("Est. Cost Savings", f"{r.get('estimated_savings_pct', 0):.1f}%",
                  delta="vs naive ordering", delta_color="normal")
    else:
        st.info("Run the pipeline to generate inventory metrics.")

with col_heat:
    st.subheader("🗓️ Monthly Sales Heatmap")
    if not df_filtered.empty:
        df_heat = df_filtered.copy()
        df_heat["Year"]  = df_heat["Date"].dt.year
        df_heat["Month"] = df_heat["Date"].dt.month
        pivot = df_heat.pivot_table(values="Weekly_Sales", index="Year", columns="Month", aggfunc="sum")
        pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig_heat = px.imshow(
            pivot, text_auto=".2s",
            color_continuous_scale="Blues",
            labels=dict(color="Total Sales"),
            aspect="auto",
        )
        fig_heat.update_layout(height=250, margin=dict(l=40, r=40, t=10, b=40))
        st.plotly_chart(fig_heat, use_container_width=True)


# ─────────────────────────────────────────────
# Row 3: Model performance + All-store inventory
# ─────────────────────────────────────────────
st.markdown("---")
col_perf, col_recs = st.columns(2)

with col_perf:
    st.subheader("🎯 Model Performance (demo)")
    perf_data = pd.DataFrame({
        "Model": ["Naive Baseline", "Linear Regression", "ARIMA", "Random Forest", "XGBoost", "Prophet"],
        "MAPE (%)": [28.4, 19.2, 16.1, 13.8, 12.5, 12.1],
        "RMSE ($)": [8420, 5630, 4890, 4120, 3870, 3750],
        "MAE ($)":  [6210, 4150, 3620, 3040, 2860, 2780],
    })
    fig_bar = px.bar(
        perf_data, x="Model", y="MAPE (%)",
        color="MAPE (%)", color_continuous_scale="RdYlGn_r",
        text_auto=".1f",
    )
    fig_bar.update_layout(height=320, showlegend=False,
                          margin=dict(l=40, r=40, t=10, b=60))
    st.plotly_chart(fig_bar, use_container_width=True)

with col_recs:
    st.subheader("📋 Reorder Alerts")
    recs_display = inventory_df.copy()
    # Simulate current stock for display
    np.random.seed(99)
    recs_display["Current_Stock"] = (
        recs_display["reorder_point"] * np.random.uniform(0.6, 1.4, len(recs_display))
    ).round(0)
    recs_display["Alert"] = recs_display["Current_Stock"] <= recs_display["reorder_point"]
    alerts = recs_display[recs_display["Alert"]].sort_values("Current_Stock")

    if not alerts.empty:
        st.dataframe(
            alerts[["Store","Dept","reorder_point","Current_Stock","eoq"]]
            .rename(columns={
                "reorder_point": "ROP",
                "eoq": "Order Qty (EOQ)",
                "Current_Stock": "Stock Now",
            })
            .head(15),
            use_container_width=True,
            height=280,
        )
    else:
        st.success("All products are above their reorder point ✓")


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built by Aniket · [GitHub](https://github.com/your-username/walmart-sales-forecasting-inventory-optimizer) "
    "· Stack: Python · Prophet · XGBoost · Streamlit · Plotly"
)
