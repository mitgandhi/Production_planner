"""
data_loader.py
--------------
Load, clean, and engineer features from AI_DATA.CSV for clustering,
statistical analysis, and production planning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

DATA_PATH = Path(__file__).parent.parent / "Data" / "AI_DATA.CSV"


# ---------------------------------------------------------------------------
# Raw loading
# ---------------------------------------------------------------------------

def load_raw(path: Optional[str] = None) -> pd.DataFrame:
    """Load the CSV as-is and return the raw DataFrame."""
    filepath = Path(path) if path else DATA_PATH
    df = pd.read_csv(
        filepath,
        dtype={"cmonyer": str, "c_sku": str, "c_sz": str, "c_cl": str},
        low_memory=False,
    )
    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich the raw DataFrame.

    Columns added:
        month, year, date   – parsed from cmonyer (MMYYYY)
        quarter             – Q1-Q4
        c_sz_num            – numeric size
    """
    df = df.copy()

    # --- date parsing -------------------------------------------------------
    df["cmonyer"] = df["cmonyer"].str.strip().str.zfill(6)
    df["month"] = df["cmonyer"].str[:2].astype(int)
    df["year"] = df["cmonyer"].str[2:].astype(int)
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01",
        errors="coerce",
    )
    df["quarter"] = df["date"].dt.quarter

    # --- column cleanup -----------------------------------------------------
    df["c_sku"] = df["c_sku"].str.strip().str.upper()
    df["c_cl"] = df["c_cl"].str.strip().str.upper()
    df["c_sz_num"] = pd.to_numeric(df["c_sz"], errors="coerce")
    df["c_qty"] = pd.to_numeric(df["c_qty"], errors="coerce").fillna(0).astype(int)

    # --- drop unusable rows -------------------------------------------------
    df = df.dropna(subset=["date"])
    df = df[df["c_qty"] > 0]
    df = df[df["c_sku"].notna() & (df["c_sku"] != "")]

    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_and_preprocess(path: Optional[str] = None) -> pd.DataFrame:
    """Convenience: load + preprocess in one call."""
    return preprocess(load_raw(path))


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------

def monthly_sku_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Total qty per (date, SKU)."""
    return (
        df.groupby(["date", "c_sku"])["c_qty"]
        .sum()
        .reset_index()
        .sort_values(["c_sku", "date"])
    )


def annual_sku_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Total qty per (year, SKU)."""
    return (
        df.groupby(["year", "c_sku"])["c_qty"]
        .sum()
        .reset_index()
        .sort_values(["c_sku", "year"])
    )


def size_color_pivot(df: pd.DataFrame, sku: str) -> pd.DataFrame:
    """Pivot table: sizes × colors for a given SKU."""
    sub = df[df["c_sku"] == sku]
    return sub.pivot_table(index="c_sz_num", columns="c_cl", values="c_qty",
                           aggfunc="sum", fill_value=0)


# ---------------------------------------------------------------------------
# Feature matrix for clustering
# ---------------------------------------------------------------------------

def build_sku_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-SKU feature matrix suitable for clustering.

    Features:
        total_qty           overall volume
        avg_monthly_qty     average monthly demand
        std_monthly_qty     standard deviation of monthly demand
        cv                  coefficient of variation
        qty_trend           linear regression slope over time
        recent_trend_pct    growth % in latest 12 months vs prior 12 months
        size_diversity      number of unique sizes ordered
        color_diversity     number of unique colors ordered
        active_months       months with non-zero demand
        peak_qty            maximum monthly qty
        seasonality_index   peak / avg  (>1 = seasonal)
    """
    monthly = monthly_sku_agg(df)
    records = []

    for sku, grp in monthly.groupby("c_sku"):
        grp = grp.sort_values("date")
        qty = grp["c_qty"].values

        # trend
        if len(qty) > 1:
            x = np.arange(len(qty))
            slope = float(np.polyfit(x, qty, 1)[0])
        else:
            slope = 0.0

        # recent trend
        n = len(qty)
        if n >= 24:
            recent_avg = qty[-12:].mean()
            prior_avg = qty[-24:-12].mean()
        elif n >= 12:
            mid = n // 2
            recent_avg = qty[mid:].mean()
            prior_avg = qty[:mid].mean()
        else:
            recent_avg = prior_avg = qty.mean()
        recent_trend_pct = (recent_avg - prior_avg) / (prior_avg + 1e-9) * 100

        avg = float(qty.mean()) if len(qty) else 0
        std = float(qty.std()) if len(qty) > 1 else 0
        cv = std / avg if avg > 0 else 0

        sku_df = df[df["c_sku"] == sku]

        records.append(
            {
                "c_sku": sku,
                "total_qty": int(qty.sum()),
                "avg_monthly_qty": round(avg, 2),
                "std_monthly_qty": round(std, 2),
                "cv": round(cv, 4),
                "qty_trend": round(slope, 4),
                "recent_trend_pct": round(recent_trend_pct, 2),
                "size_diversity": int(sku_df["c_sz_num"].nunique()),
                "color_diversity": int(sku_df["c_cl"].nunique()),
                "active_months": int(n),
                "peak_qty": int(qty.max()),
                "seasonality_index": round(float(qty.max()) / (avg + 1e-9), 4),
            }
        )

    feat_df = pd.DataFrame(records).set_index("c_sku")
    return feat_df


# ---------------------------------------------------------------------------
# Quick summary helpers
# ---------------------------------------------------------------------------

def summary_stats(df: pd.DataFrame) -> dict:
    """High-level dataset summary for dashboard KPIs."""
    return {
        "total_records": len(df),
        "unique_skus": df["c_sku"].nunique(),
        "unique_sizes": df["c_sz_num"].nunique(),
        "unique_colors": df["c_cl"].nunique(),
        "total_units": int(df["c_qty"].sum()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "years_covered": int(df["year"].max() - df["year"].min() + 1),
    }
