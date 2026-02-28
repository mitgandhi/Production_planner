"""
plots.py
--------
All matplotlib/seaborn plot factory functions.
Each function returns a matplotlib Figure so it can be embedded in PyQt6
or saved to disk.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # off-screen rendering; Qt backend is set per-canvas
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# Global style
sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams.update({"figure.dpi": 100, "font.size": 9})

_PALETTE = sns.color_palette("tab10", 20)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _new_fig(w: float = 10, h: float = 6) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def _new_figs(rows: int, cols: int, w: float = 14, h: float = 8):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    return fig, axes


# ---------------------------------------------------------------------------
# Dashboard charts
# ---------------------------------------------------------------------------

def plot_monthly_total(df: pd.DataFrame) -> plt.Figure:
    """Overall monthly units shipped."""
    monthly = df.groupby("date")["c_qty"].sum().reset_index()
    fig, ax = _new_fig(12, 5)
    ax.fill_between(monthly["date"], monthly["c_qty"], alpha=0.3, color="#1f77b4")
    ax.plot(monthly["date"], monthly["c_qty"], color="#1f77b4", lw=1.5)
    ax.set_title("Monthly Total Units", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Units")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    return fig


def plot_top_skus(df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """Horizontal bar chart of top-N SKUs by total volume."""
    top = (
        df.groupby("c_sku")["c_qty"]
        .sum()
        .nlargest(top_n)
        .sort_values()
    )
    fig, ax = _new_fig(10, 6)
    colors = _PALETTE[: len(top)]
    bars = ax.barh(top.index, top.values, color=colors)
    ax.bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=8)
    ax.set_title(f"Top {top_n} SKUs by Volume", fontsize=13, fontweight="bold")
    ax.set_xlabel("Total Units")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    return fig


def plot_size_distribution(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of demand across sizes."""
    size_qty = (
        df.groupby("c_sz_num")["c_qty"]
        .sum()
        .dropna()
        .sort_index()
    )
    fig, ax = _new_fig(10, 5)
    ax.bar(size_qty.index.astype(str), size_qty.values, color="#2ca02c", edgecolor="white")
    ax.set_title("Demand by Size", fontsize=13, fontweight="bold")
    ax.set_xlabel("Size")
    ax.set_ylabel("Total Units")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    return fig


def plot_color_distribution(df: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    """Bar chart of top-N colors by demand."""
    color_qty = (
        df.groupby("c_cl")["c_qty"]
        .sum()
        .nlargest(top_n)
        .sort_values(ascending=False)
    )
    fig, ax = _new_fig(12, 5)
    ax.bar(range(len(color_qty)), color_qty.values,
           color=_PALETTE[: len(color_qty)], edgecolor="white")
    ax.set_xticks(range(len(color_qty)))
    ax.set_xticklabels(color_qty.index, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"Top {top_n} Colors by Volume", fontsize=13, fontweight="bold")
    ax.set_ylabel("Total Units")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    return fig


def plot_annual_trend(df: pd.DataFrame) -> plt.Figure:
    """Annual units bar chart with YoY growth annotation."""
    annual = df.groupby("year")["c_qty"].sum().reset_index()
    fig, ax = _new_fig(10, 5)
    bars = ax.bar(annual["year"].astype(str), annual["c_qty"],
                  color="#9467bd", edgecolor="white")
    ax.bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=8)
    ax.set_title("Annual Total Units", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Units")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    return fig


def plot_heatmap_sku_month(df: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    """Heatmap: top-N SKUs × month-of-year demand."""
    top_skus = df.groupby("c_sku")["c_qty"].sum().nlargest(top_n).index
    sub = df[df["c_sku"].isin(top_skus)].copy()
    sub["month_num"] = sub["date"].dt.month
    pivot = sub.pivot_table(index="c_sku", columns="month_num", values="c_qty",
                            aggfunc="sum", fill_value=0)
    pivot.columns = [f"M{c:02d}" for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", fmt=",", annot=False,
                linewidths=0.3, cbar_kws={"label": "Units"})
    ax.set_title(f"Monthly Demand Heatmap – Top {top_n} SKUs", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("SKU")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Clustering plots
# ---------------------------------------------------------------------------

def plot_cluster_scatter_2d(
    coords: np.ndarray,
    labels: np.ndarray,
    sku_names: List[str],
    variance_ratio: np.ndarray,
    label_map: Optional[Dict[int, str]] = None,
    title: str = "Cluster Scatter (PCA)",
) -> plt.Figure:
    """2-D PCA scatter coloured by cluster."""
    unique_labels = sorted(set(labels))
    cmap = cm.get_cmap("tab10", len(unique_labels))
    fig, ax = _new_fig(11, 7)

    for i, cid in enumerate(unique_labels):
        mask = labels == cid
        colour = "grey" if cid == -1 else cmap(i)
        display = "Noise" if cid == -1 else (label_map.get(cid, f"Cluster {cid}") if label_map else f"Cluster {cid}")
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[colour], label=display, s=60, alpha=0.8, edgecolors="white", lw=0.4)

    ax.set_xlabel(f"PC1 ({variance_ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({variance_ratio[1]*100:.1f}%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_elbow(k_range: List[int], inertias: List[float], silhouettes: List[float]) -> plt.Figure:
    """Elbow + silhouette side-by-side to pick optimal K."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(k_range, inertias, "bo-", lw=1.5)
    ax1.set_title("Elbow Method (Inertia)", fontweight="bold")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia")

    ax2.plot(k_range, silhouettes, "gs-", lw=1.5)
    ax2.set_title("Silhouette Score", fontweight="bold")
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette Score")

    fig.suptitle("Optimal K Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_cluster_profiles(result_df: pd.DataFrame) -> plt.Figure:
    """Radar-like bar comparison of cluster centres."""
    numeric_cols = ["avg_monthly_qty", "cv", "qty_trend", "size_diversity",
                    "color_diversity", "seasonality_index"]
    cols = [c for c in numeric_cols if c in result_df.columns]
    if not cols:
        fig, ax = _new_fig()
        ax.text(0.5, 0.5, "No numeric features available", ha="center")
        return fig

    cluster_means = result_df.groupby("cluster_label")[cols].mean()
    # Normalise to [0,1] for comparison
    normed = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(cols))
    width = 0.8 / len(normed)
    for i, (lbl, row) in enumerate(normed.iterrows()):
        ax.bar(x + i * width, row.values, width, label=lbl, alpha=0.85)

    ax.set_xticks(x + width * (len(normed) - 1) / 2)
    ax.set_xticklabels(cols, rotation=20, ha="right", fontsize=8)
    ax.set_title("Cluster Feature Profiles (Normalised)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Normalised Value")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_abc_xyz_matrix(result_df: pd.DataFrame) -> plt.Figure:
    """Count heatmap of ABC × XYZ matrix."""
    if "abc" not in result_df.columns or "xyz" not in result_df.columns:
        fig, ax = _new_fig()
        ax.text(0.5, 0.5, "ABC/XYZ not computed", ha="center")
        return fig

    pivot = result_df.groupby(["abc", "xyz"]).size().unstack("xyz", fill_value=0)
    for col in ["X", "Y", "Z"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["X", "Y", "Z"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(pivot, ax=ax, annot=True, fmt="d", cmap="Blues",
                linewidths=0.5, cbar_kws={"label": "SKU count"})
    ax.set_title("ABC × XYZ Matrix (SKU Count)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Demand Variability (XYZ)")
    ax.set_ylabel("Volume Class (ABC)")
    fig.tight_layout()
    return fig


def plot_cluster_volume_pie(result_df: pd.DataFrame) -> plt.Figure:
    """Pie chart of total volume per cluster label."""
    vol = result_df.groupby("cluster_label")["total_qty"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    wedge_props = {"linewidth": 1, "edgecolor": "white"}
    ax.pie(vol.values, labels=vol.index, autopct="%1.1f%%",
           startangle=140, wedgeprops=wedge_props,
           colors=_PALETTE[: len(vol)])
    ax.set_title("Volume Share by Cluster", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Statistical analysis plots
# ---------------------------------------------------------------------------

def plot_distribution(df: pd.DataFrame, sku: str) -> plt.Figure:
    """Histogram + KDE for a single SKU's monthly demand."""
    monthly = df[df["c_sku"] == sku].groupby("date")["c_qty"].sum()
    fig, ax = _new_fig(9, 5)
    sns.histplot(monthly, ax=ax, kde=True, color="#1f77b4", edgecolor="white")
    ax.set_title(f"Monthly Demand Distribution – {sku}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Monthly Quantity")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> plt.Figure:
    """Heatmap of SKU correlation matrix."""
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, ax=ax, mask=mask, cmap="coolwarm",
                vmin=-1, vmax=1, annot=False, square=True,
                linewidths=0.1, cbar_kws={"label": "Pearson r"})
    ax.set_title("SKU Demand Correlation Matrix", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    return fig


def plot_seasonality_heatmap(seasonality_df: pd.DataFrame, top_n: int = 25) -> plt.Figure:
    """Heatmap of seasonal indices per SKU."""
    month_cols = [c for c in seasonality_df.columns if c.startswith("M")]
    top_skus = seasonality_df["seasonality_strength"].nlargest(top_n).index
    sub = seasonality_df.loc[top_skus, month_cols].fillna(1.0)
    sub.columns = [f"{'JanFebMarAprMayJunJulAugSepOctNovDec'.split()[i-1] if False else c}" for i, c in enumerate(sub.columns, 1)]

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(sub, ax=ax, cmap="RdYlGn", center=1.0, annot=True, fmt=".2f",
                linewidths=0.3, cbar_kws={"label": "Seasonal Index"}, annot_kws={"size": 7})
    ax.set_title(f"Seasonal Index Heatmap – Top {top_n} SKUs (by strength)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("SKU")
    fig.tight_layout()
    return fig


def plot_trend_overview(trend_df: pd.DataFrame) -> plt.Figure:
    """Bar chart of slope per SKU coloured by direction."""
    top = trend_df.nlargest(20, "Total") if "Total" in trend_df.columns else trend_df.head(20)
    top = trend_df.sort_values("slope").head(40)
    colors = top["direction"].map({"Uptrend": "#2ca02c", "Downtrend": "#d62728", "Flat": "#7f7f7f"})
    fig, ax = _new_fig(12, 7)
    ax.barh(top.index, top["slope"], color=colors.values)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Demand Trend Slope by SKU", fontsize=13, fontweight="bold")
    ax.set_xlabel("Slope (units/month)")
    fig.tight_layout()
    return fig


def plot_outlier_timeline(outlier_df: pd.DataFrame) -> plt.Figure:
    """Scatter plot of outlier records on a timeline."""
    if outlier_df.empty:
        fig, ax = _new_fig()
        ax.text(0.5, 0.5, "No outliers detected", ha="center", va="center", fontsize=12)
        return fig
    fig, ax = _new_fig(12, 5)
    ax.scatter(outlier_df["date"], outlier_df["c_qty"], c="red", s=25, alpha=0.7)
    ax.set_title("Outlier Records Timeline", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Quantity")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Production planning plots
# ---------------------------------------------------------------------------

def plot_sku_forecast(
    history_dates: pd.Series,
    history_qty: np.ndarray,
    forecast_dates: pd.Series,
    forecast_qty: np.ndarray,
    sku: str,
    ci_lower: Optional[np.ndarray] = None,
    ci_upper: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Single SKU demand + forecast plot with optional confidence interval."""
    fig, ax = _new_fig(12, 5)
    ax.plot(history_dates, history_qty, "b-o", ms=3, lw=1.5, label="Historical")
    ax.plot(forecast_dates, forecast_qty, "r--s", ms=4, lw=1.5, label="Forecast")
    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(forecast_dates, ci_lower, ci_upper, alpha=0.2, color="red", label="95% CI")
    ax.axvline(history_dates.iloc[-1], color="grey", ls=":", lw=1)
    ax.set_title(f"Demand Forecast – {sku}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Units")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    return fig


def plot_safety_stock_comparison(plan_df: pd.DataFrame) -> plt.Figure:
    """Bar chart comparing safety stock vs recommended production per SKU."""
    top = plan_df.nlargest(20, "avg_monthly_qty") if len(plan_df) > 20 else plan_df
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(top))
    w = 0.35
    ax.bar(x - w / 2, top["avg_monthly_qty"], w, label="Avg Monthly Demand", color="#1f77b4")
    ax.bar(x + w / 2, top["safety_stock"], w, label="Safety Stock", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(top["c_sku"], rotation=45, ha="right", fontsize=8)
    ax.set_title("Avg Monthly Demand vs Safety Stock (Top 20 SKUs)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Units")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    return fig


def plot_production_plan_gantt(plan_df: pd.DataFrame, months: int = 6) -> plt.Figure:
    """Simple production schedule as a coloured table/chart."""
    top = plan_df.nlargest(15, "avg_monthly_qty")
    month_labels = [f"M+{i+1}" for i in range(months)]
    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = cm.get_cmap("YlOrRd")

    for i, (_, row) in enumerate(top.iterrows()):
        for j, m in enumerate(month_labels):
            col_key = f"forecast_m{j+1}"
            val = float(row.get(col_key, row.get("avg_monthly_qty", 0)))
            norm_val = val / (plan_df["avg_monthly_qty"].max() + 1e-9)
            rect = plt.Rectangle([j, i], 1, 0.85, color=cmap(norm_val))
            ax.add_patch(rect)
            ax.text(j + 0.5, i + 0.42, f"{val:,.0f}", ha="center", va="center",
                    fontsize=7, color="black" if norm_val < 0.6 else "white")

    ax.set_xlim(0, months)
    ax.set_ylim(0, len(top))
    ax.set_xticks(np.arange(months) + 0.5)
    ax.set_xticklabels(month_labels)
    ax.set_yticks(np.arange(len(top)) + 0.42)
    ax.set_yticklabels(top["c_sku"].tolist() if "c_sku" in top.columns else top.index.tolist(), fontsize=8)
    ax.set_title("Production Plan – Forecast Units (Top 15 SKUs)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Planning Month")
    ax.set_ylabel("SKU")
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Relative Demand")
    fig.tight_layout()
    return fig
