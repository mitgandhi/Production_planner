"""
production_planning.py
-----------------------
Demand forecasting, safety-stock calculation, reorder-point logic,
and ABC-XYZ-based production scheduling for the apparel dataset.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _monthly_series(df: pd.DataFrame, sku: str) -> pd.Series:
    """Return a date-indexed monthly qty series for one SKU."""
    sub = df[df["c_sku"] == sku].groupby("date")["c_qty"].sum().sort_index()
    return sub


# ---------------------------------------------------------------------------
# Forecasting methods
# ---------------------------------------------------------------------------

def moving_average_forecast(
    series: pd.Series, horizon: int = 6, window: int = 3
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Simple moving average forecast."""
    last_val = float(series.rolling(window, min_periods=1).mean().iloc[-1])
    forecast = np.full(horizon, last_val)
    last_date = series.index[-1]
    future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=horizon, freq="MS")
    return forecast, future_dates


def exponential_smoothing_forecast(
    series: pd.Series, horizon: int = 6, alpha: float = 0.3
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Holt-Winters simple exponential smoothing (manual)."""
    qty = series.values.astype(float)
    level = qty[0]
    for v in qty[1:]:
        level = alpha * v + (1 - alpha) * level
    forecast = np.full(horizon, level)
    future_dates = pd.date_range(
        series.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS"
    )
    return forecast, future_dates


def linear_trend_forecast(
    series: pd.Series, horizon: int = 6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Linear regression forecast with 95% confidence interval."""
    qty = series.values.astype(float)
    x = np.arange(len(qty))
    coeffs = np.polyfit(x, qty, 1)
    poly = np.poly1d(coeffs)

    x_future = np.arange(len(qty), len(qty) + horizon)
    forecast = poly(x_future)
    forecast = np.clip(forecast, 0, None)

    residuals = qty - poly(x)
    std_err = residuals.std()
    ci_lower = np.clip(forecast - 1.96 * std_err, 0, None)
    ci_upper = forecast + 1.96 * std_err

    future_dates = pd.date_range(
        series.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS"
    )
    return forecast, ci_lower, ci_upper, future_dates


def seasonal_naive_forecast(
    series: pd.Series, horizon: int = 6
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Seasonal naïve: repeat the same month from last year."""
    qty = series.values.astype(float)
    period = 12
    forecast = np.array([
        qty[-(period - (i % period))] if (period - (i % period)) <= len(qty) else qty.mean()
        for i in range(horizon)
    ])
    forecast = np.clip(forecast, 0, None)
    future_dates = pd.date_range(
        series.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS"
    )
    return forecast, future_dates


def weighted_ensemble_forecast(
    series: pd.Series, horizon: int = 6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Ensemble of MA + ES + linear trend (equal weights).
    Also returns ±1 std as a lightweight confidence interval.
    """
    f_ma, dates = moving_average_forecast(series, horizon)
    f_es, _ = exponential_smoothing_forecast(series, horizon)
    f_lt, ci_lo, ci_hi, _ = linear_trend_forecast(series, horizon)
    f_sn, _ = seasonal_naive_forecast(series, horizon)

    ensemble = np.mean([f_ma, f_es, f_lt, f_sn], axis=0)
    std = np.std([f_ma, f_es, f_lt, f_sn], axis=0)
    ci_lower = np.clip(ensemble - 1.96 * std, 0, None)
    ci_upper = ensemble + 1.96 * std
    return ensemble, ci_lower, ci_upper, dates


# ---------------------------------------------------------------------------
# Safety stock & reorder point
# ---------------------------------------------------------------------------

def compute_safety_stock(
    avg_demand: float,
    std_demand: float,
    lead_time_months: float = 1.0,
    service_level: float = 0.95,
) -> float:
    """
    Safety Stock = Z * σ_d * √LT
    where Z is the service-level z-score.
    """
    z = norm.ppf(service_level)
    return max(0.0, z * std_demand * np.sqrt(lead_time_months))


def compute_reorder_point(
    avg_demand: float,
    lead_time_months: float,
    safety_stock: float,
) -> float:
    """ROP = avg_demand * lead_time + safety_stock"""
    return avg_demand * lead_time_months + safety_stock


# ---------------------------------------------------------------------------
# Full planning table
# ---------------------------------------------------------------------------

def build_production_plan(
    df: pd.DataFrame,
    horizon: int = 6,
    lead_time_months: float = 1.0,
    service_level: float = 0.95,
    method: str = "ensemble",
) -> pd.DataFrame:
    """
    Build a production plan for all SKUs.

    Returns a DataFrame with one row per SKU containing:
        avg_monthly_qty, std_monthly_qty, safety_stock,
        reorder_point, forecast_m1…forecast_m{horizon}
    """
    records = []

    for sku in df["c_sku"].unique():
        series = _monthly_series(df, sku)
        if len(series) < 2:
            continue

        avg = float(series.mean())
        std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
        ss = compute_safety_stock(avg, std, lead_time_months, service_level)
        rop = compute_reorder_point(avg, lead_time_months, ss)

        if method == "moving_average":
            forecast, dates = moving_average_forecast(series, horizon)
            ci_lo = ci_hi = None
        elif method == "linear":
            forecast, ci_lo, ci_hi, dates = linear_trend_forecast(series, horizon)
        elif method == "seasonal":
            forecast, dates = seasonal_naive_forecast(series, horizon)
            ci_lo = ci_hi = None
        else:  # ensemble (default)
            forecast, ci_lo, ci_hi, dates = weighted_ensemble_forecast(series, horizon)

        row: Dict = {
            "c_sku": sku,
            "active_months": len(series),
            "avg_monthly_qty": round(avg, 1),
            "std_monthly_qty": round(std, 1),
            "safety_stock": round(ss, 1),
            "reorder_point": round(rop, 1),
            "total_planned": round(float(forecast.sum()), 1),
        }
        for i, (fv, d) in enumerate(zip(forecast, dates), 1):
            row[f"forecast_m{i}"] = round(float(max(fv, 0)), 1)
            row[f"forecast_date_m{i}"] = str(d.date())

        records.append(row)

    plan_df = pd.DataFrame(records)
    return plan_df


# ---------------------------------------------------------------------------
# Current Production Estimate  (per SKU × Size × Color)
# ---------------------------------------------------------------------------

def current_production_estimate(df: pd.DataFrame, sales_days: int = 90, production_days: int = 45) -> pd.DataFrame:
    """
    For every SKU + Size + Color combination calculate:
        PerDaySalesQty   = Total Sales (last `sales_days` days) / sales_days
        ProductionReqQty = PerDaySalesQty × production_days

    Parameters
    ----------
    df              : preprocessed DataFrame (must have 'date', 'c_sku', 'c_sz', 'c_cl', 'c_qty')
    sales_days      : lookback window in days (default 90)
    production_days : forward production horizon in days (default 45)
    """
    max_date = df["date"].max()
    cutoff   = max_date - pd.Timedelta(days=sales_days)

    recent = df[df["date"] > cutoff]

    agg = (
        recent
        .groupby(["c_sku", "c_sz", "c_cl"], as_index=False)["c_qty"]
        .sum()
        .rename(columns={"c_qty": "total_sales_90d"})
    )

    agg["per_day_sales_qty"]  = (agg["total_sales_90d"] / sales_days).round(2)
    agg["production_req_qty"] = (agg["per_day_sales_qty"] * production_days).round(0).astype(int)

    # tidy column order & sort by production requirement descending
    agg = agg[["c_sku", "c_sz", "c_cl",
               "total_sales_90d", "per_day_sales_qty", "production_req_qty"]]
    agg = agg.sort_values("production_req_qty", ascending=False).reset_index(drop=True)

    return agg


# ---------------------------------------------------------------------------
# Summary recommendations (text)
# ---------------------------------------------------------------------------

def generate_recommendations(
    plan_df: pd.DataFrame,
    cluster_result: Optional[pd.DataFrame] = None,
) -> List[Dict]:
    """
    Generate plain-language production recommendations.
    """
    recs = []

    # Top 5 volume drivers
    top5 = plan_df.nlargest(5, "avg_monthly_qty")
    for _, row in top5.iterrows():
        recs.append(
            {
                "sku": row["c_sku"],
                "priority": "HIGH",
                "action": "Maintain Production",
                "reason": f"High-volume SKU averaging {row['avg_monthly_qty']:,.0f} units/month.",
                "safety_stock": row["safety_stock"],
                "reorder_point": row["reorder_point"],
                "next_6_months_forecast": row["total_planned"],
            }
        )

    # Growing SKUs (positive ensemble trend)
    if cluster_result is not None and "cluster_label" in cluster_result.columns:
        growing = cluster_result[
            cluster_result["cluster_label"].str.contains("Growing", na=False)
        ]["c_sku"].tolist()
        for sku in growing[:5]:
            sub = plan_df[plan_df["c_sku"] == sku]
            if sub.empty:
                continue
            row = sub.iloc[0]
            recs.append(
                {
                    "sku": sku,
                    "priority": "MEDIUM",
                    "action": "Increase Production",
                    "reason": "Cluster analysis identified positive demand trend.",
                    "safety_stock": row["safety_stock"],
                    "reorder_point": row["reorder_point"],
                    "next_6_months_forecast": row["total_planned"],
                }
            )

    # Declining / Volatile SKUs
    if cluster_result is not None and "cluster_label" in cluster_result.columns:
        declining = cluster_result[
            cluster_result["cluster_label"].str.contains("Declining|Volatile", na=False)
        ]["c_sku"].tolist()
        for sku in declining[:5]:
            sub = plan_df[plan_df["c_sku"] == sku]
            if sub.empty:
                continue
            row = sub.iloc[0]
            recs.append(
                {
                    "sku": sku,
                    "priority": "LOW",
                    "action": "Review / Reduce",
                    "reason": "Declining or highly volatile demand. Consider phasing out or reducing batch size.",
                    "safety_stock": row["safety_stock"],
                    "reorder_point": row["reorder_point"],
                    "next_6_months_forecast": row["total_planned"],
                }
            )

    return recs
