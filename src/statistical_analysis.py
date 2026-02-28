"""
statistical_analysis.py
------------------------
Statistical preprocessing and analysis of the apparel inventory dataset.
Produces summary tables, distribution diagnostics, seasonality metrics,
and a structured JSON-ready context for passing to Qwen 3.
"""

from __future__ import annotations

import json
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import shapiro, kstest, anderson

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-SKU descriptive statistics table."""
    monthly = (
        df.groupby(["date", "c_sku"])["c_qty"]
        .sum()
        .reset_index()
    )
    records = []
    for sku, grp in monthly.groupby("c_sku"):
        qty = grp["c_qty"].values
        records.append(
            {
                "SKU": sku,
                "N_months": len(qty),
                "Total": int(qty.sum()),
                "Mean": round(qty.mean(), 2),
                "Median": round(float(np.median(qty)), 2),
                "Std": round(qty.std(ddof=1) if len(qty) > 1 else 0, 2),
                "CV%": round(qty.std(ddof=1) / qty.mean() * 100 if qty.mean() > 0 else 0, 1),
                "Min": int(qty.min()),
                "Max": int(qty.max()),
                "Q1": round(float(np.percentile(qty, 25)), 2),
                "Q3": round(float(np.percentile(qty, 75)), 2),
                "IQR": round(float(np.percentile(qty, 75) - np.percentile(qty, 25)), 2),
                "Skewness": round(float(sp_stats.skew(qty)), 4),
                "Kurtosis": round(float(sp_stats.kurtosis(qty)), 4),
            }
        )
    return pd.DataFrame(records).set_index("SKU")


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def detect_outliers(df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
    """
    Identify outlier records.

    Parameters
    ----------
    method : 'iqr' or 'zscore'
    """
    monthly = df.groupby(["date", "c_sku"])["c_qty"].sum().reset_index()

    results = []
    for sku, grp in monthly.groupby("c_sku"):
        qty = grp["c_qty"].values
        if method == "iqr":
            q1, q3 = np.percentile(qty, 25), np.percentile(qty, 75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (qty < lo) | (qty > hi)
        else:
            z = np.abs(sp_stats.zscore(qty))
            mask = z > 3

        outlier_rows = grp[mask].copy()
        outlier_rows["c_sku"] = sku
        outlier_rows["outlier_method"] = method
        results.append(outlier_rows)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame(columns=["date", "c_sku", "c_qty", "outlier_method"])


# ---------------------------------------------------------------------------
# Normality tests
# ---------------------------------------------------------------------------

def normality_tests(df: pd.DataFrame, max_skus: int = 50) -> pd.DataFrame:
    """
    Run Shapiro-Wilk and K-S normality tests per SKU (top N by volume).
    """
    monthly = df.groupby(["date", "c_sku"])["c_qty"].sum().reset_index()
    top_skus = (
        monthly.groupby("c_sku")["c_qty"].sum()
        .nlargest(max_skus)
        .index.tolist()
    )

    records = []
    for sku in top_skus:
        qty = monthly[monthly["c_sku"] == sku]["c_qty"].values
        if len(qty) < 3:
            continue
        try:
            sw_stat, sw_p = shapiro(qty)
        except Exception:
            sw_stat, sw_p = np.nan, np.nan
        try:
            ks_stat, ks_p = kstest(qty, "norm", args=(qty.mean(), qty.std()))
        except Exception:
            ks_stat, ks_p = np.nan, np.nan

        records.append(
            {
                "SKU": sku,
                "SW_stat": round(float(sw_stat), 4),
                "SW_p": round(float(sw_p), 4),
                "SW_normal": sw_p > 0.05,
                "KS_stat": round(float(ks_stat), 4),
                "KS_p": round(float(ks_p), 4),
                "KS_normal": ks_p > 0.05,
            }
        )
    return pd.DataFrame(records).set_index("SKU")


# ---------------------------------------------------------------------------
# Seasonality metrics
# ---------------------------------------------------------------------------

def seasonality_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly seasonal index per SKU:
        SI(m) = avg_qty_month_m / overall_monthly_avg
    Values > 1 indicate peak demand months.
    """
    monthly = df.groupby(["date", "c_sku"])["c_qty"].sum().reset_index()
    monthly["month_num"] = monthly["date"].dt.month

    records = []
    for sku, grp in monthly.groupby("c_sku"):
        overall_avg = grp["c_qty"].mean()
        if overall_avg == 0:
            continue
        month_avg = grp.groupby("month_num")["c_qty"].mean()
        si = (month_avg / overall_avg).round(3)
        row = {"SKU": sku}
        row.update({f"M{m:02d}": si.get(m, np.nan) for m in range(1, 13)})
        row["peak_month"] = int(si.idxmax()) if not si.isna().all() else 0
        row["trough_month"] = int(si.idxmin()) if not si.isna().all() else 0
        row["seasonality_strength"] = round(float(si.max() - si.min()), 4)
        records.append(row)

    return pd.DataFrame(records).set_index("SKU")


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------

def trend_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Linear trend slope, R², and classification per SKU.
    """
    monthly = df.groupby(["date", "c_sku"])["c_qty"].sum().reset_index()
    records = []
    for sku, grp in monthly.groupby("c_sku"):
        grp = grp.sort_values("date")
        qty = grp["c_qty"].values
        n = len(qty)
        if n < 3:
            continue
        x = np.arange(n)
        slope, intercept, r, p, se = sp_stats.linregress(x, qty)
        r2 = r ** 2

        if slope > 1 and p < 0.05:
            direction = "Uptrend"
        elif slope < -1 and p < 0.05:
            direction = "Downtrend"
        else:
            direction = "Flat"

        records.append(
            {
                "SKU": sku,
                "slope": round(float(slope), 4),
                "intercept": round(float(intercept), 2),
                "R2": round(float(r2), 4),
                "p_value": round(float(p), 4),
                "significant": p < 0.05,
                "direction": direction,
            }
        )
    return pd.DataFrame(records).set_index("SKU")


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def sku_correlation_matrix(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """
    Pearson correlation matrix of monthly demand for top-N SKUs.
    """
    monthly = df.groupby(["date", "c_sku"])["c_qty"].sum().unstack("c_sku").fillna(0)
    top_skus = monthly.sum().nlargest(top_n).index
    return monthly[top_skus].corr()


# ---------------------------------------------------------------------------
# Preprocessing report for Qwen 3
# ---------------------------------------------------------------------------

def build_qwen_context(
    df: pd.DataFrame,
    desc_stats: Optional[pd.DataFrame] = None,
    seasonality: Optional[pd.DataFrame] = None,
    trend: Optional[pd.DataFrame] = None,
    top_n_skus: int = 20,
) -> dict:
    """
    Build a structured dictionary (JSON-serialisable) that summarises the
    dataset and analysis results.  Pass this as system context to Qwen 3.
    """
    if desc_stats is None:
        desc_stats = descriptive_stats(df)
    if seasonality is None:
        seasonality = seasonality_profile(df)
    if trend is None:
        trend = trend_analysis(df)

    top_skus_by_vol = desc_stats["Total"].nlargest(top_n_skus).index.tolist()

    context = {
        "dataset_summary": {
            "total_records": len(df),
            "unique_skus": int(df["c_sku"].nunique()),
            "unique_sizes": int(df["c_sz_num"].nunique()),
            "unique_colors": int(df["c_cl"].nunique()),
            "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
            "total_units": int(df["c_qty"].sum()),
        },
        "top_skus_by_volume": top_skus_by_vol,
        "sku_stats": {},
        "seasonality_highlights": {},
        "trend_summary": {
            "uptrend_skus": trend[trend["direction"] == "Uptrend"].index.tolist(),
            "downtrend_skus": trend[trend["direction"] == "Downtrend"].index.tolist(),
            "flat_skus": trend[trend["direction"] == "Flat"].index.tolist(),
        },
    }

    for sku in top_skus_by_vol:
        row = desc_stats.loc[sku] if sku in desc_stats.index else {}
        s_row = seasonality.loc[sku] if sku in seasonality.index else {}
        t_row = trend.loc[sku] if sku in trend.index else {}

        context["sku_stats"][sku] = {
            "total_units": int(row.get("Total", 0)),
            "avg_monthly": float(row.get("Mean", 0)),
            "cv_pct": float(row.get("CV%", 0)),
            "trend_direction": str(t_row.get("direction", "Unknown")),
            "trend_slope": float(t_row.get("slope", 0)),
            "peak_month": int(s_row.get("peak_month", 0)),
            "seasonality_strength": float(s_row.get("seasonality_strength", 0)),
        }

    return context


def export_qwen_context_json(context: dict, path: str) -> None:
    """Write the context dict to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Normalisation helpers (for ML pipelines)
# ---------------------------------------------------------------------------

def normalise_monthly_series(df: pd.DataFrame, method: str = "minmax") -> pd.DataFrame:
    """
    Normalise each SKU's monthly qty series independently.

    method: 'minmax' | 'zscore'
    """
    monthly = df.groupby(["date", "c_sku"])["c_qty"].sum().reset_index()
    normed_frames = []

    for sku, grp in monthly.groupby("c_sku"):
        grp = grp.copy().sort_values("date")
        qty = grp["c_qty"].values.astype(float)
        if method == "minmax":
            lo, hi = qty.min(), qty.max()
            grp["qty_norm"] = (qty - lo) / (hi - lo + 1e-9)
        else:
            grp["qty_norm"] = (qty - qty.mean()) / (qty.std(ddof=1) + 1e-9)
        normed_frames.append(grp)

    return pd.concat(normed_frames, ignore_index=True)
