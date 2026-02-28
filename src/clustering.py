"""
clustering.py
-------------
Clustering algorithms, ABC-XYZ partitioning, cluster labeling,
and elbow/silhouette analysis for the apparel inventory dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Feature scaling helper
# ---------------------------------------------------------------------------

CLUSTER_FEATURES = [
    "total_qty",
    "avg_monthly_qty",
    "cv",
    "qty_trend",
    "recent_trend_pct",
    "size_diversity",
    "color_diversity",
    "active_months",
    "seasonality_index",
]


def _scale(features: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    cols = [c for c in CLUSTER_FEATURES if c in features.columns]
    X = features[cols].fillna(0).values
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler, cols


# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------

def run_kmeans(
    features: pd.DataFrame, n_clusters: int = 5
) -> Tuple[np.ndarray, float, KMeans, StandardScaler]:
    X, scaler, _ = _scale(features)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels) if len(set(labels)) > 1 else 0.0
    return labels, round(score, 4), km, scaler


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------

def run_dbscan(
    features: pd.DataFrame, eps: float = 1.2, min_samples: int = 3
) -> Tuple[np.ndarray, StandardScaler]:
    X, scaler, _ = _scale(features)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    return labels, scaler


# ---------------------------------------------------------------------------
# Hierarchical / Agglomerative
# ---------------------------------------------------------------------------

def run_hierarchical(
    features: pd.DataFrame, n_clusters: int = 5, linkage: str = "ward"
) -> Tuple[np.ndarray, StandardScaler]:
    X, scaler, _ = _scale(features)
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = hc.fit_predict(X)
    return labels, scaler


# ---------------------------------------------------------------------------
# PCA projection (for 2-D / 3-D scatter plots)
# ---------------------------------------------------------------------------

def pca_coords(
    features: pd.DataFrame,
    scaler: StandardScaler | None = None,
    n_components: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    cols = [c for c in CLUSTER_FEATURES if c in features.columns]
    X = features[cols].fillna(0).values
    if scaler is not None:
        X = scaler.transform(X)
    else:
        X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_


# ---------------------------------------------------------------------------
# Elbow / silhouette search
# ---------------------------------------------------------------------------

def find_optimal_k(
    features: pd.DataFrame, max_k: int = 12
) -> Tuple[List[int], List[float], List[float]]:
    X, _, _ = _scale(features)
    k_range = list(range(2, min(max_k + 1, len(features))))
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbls = km.fit_predict(X)
        inertias.append(float(km.inertia_))
        sil = silhouette_score(X, lbls) if len(set(lbls)) > 1 else 0.0
        silhouettes.append(round(sil, 4))
    return k_range, inertias, silhouettes


# ---------------------------------------------------------------------------
# ABC analysis (volume-based Pareto)
# ---------------------------------------------------------------------------

def abc_analysis(features: pd.DataFrame) -> pd.Series:
    """
    A  – top 80 % of cumulative volume
    B  – next 15 % (80–95 %)
    C  – bottom 5 % (95–100 %)
    """
    sorted_qty = features["total_qty"].sort_values(ascending=False)
    cum_share = sorted_qty.cumsum() / sorted_qty.sum()
    abc = pd.Series("C", index=sorted_qty.index, name="abc")
    abc[cum_share <= 0.80] = "A"
    abc[(cum_share > 0.80) & (cum_share <= 0.95)] = "B"
    return abc


# ---------------------------------------------------------------------------
# XYZ analysis (demand variability)
# ---------------------------------------------------------------------------

def xyz_analysis(features: pd.DataFrame) -> pd.Series:
    """
    X  – CV < 0.25  (very predictable)
    Y  – 0.25 ≤ CV < 0.50
    Z  – CV ≥ 0.50  (highly variable / erratic)
    """
    cv = features["cv"]
    xyz = pd.Series("Z", index=cv.index, name="xyz")
    xyz[cv < 0.25] = "X"
    xyz[(cv >= 0.25) & (cv < 0.50)] = "Y"
    return xyz


# ---------------------------------------------------------------------------
# Cluster label generation
# ---------------------------------------------------------------------------

def label_clusters(
    features: pd.DataFrame, labels: np.ndarray
) -> Dict[int, str]:
    """
    Assign a human-readable business label to each cluster centre
    based on volume, trend, and variability statistics.
    """
    feat = features.copy()
    feat["_cluster"] = labels

    stats = feat.groupby("_cluster").agg(
        total_qty=("total_qty", "mean"),
        trend=("qty_trend", "mean"),
        recent=("recent_trend_pct", "mean"),
        cv=("cv", "mean"),
        seasonal=("seasonality_index", "mean"),
    )

    q75 = stats["total_qty"].quantile(0.75)
    q25 = stats["total_qty"].quantile(0.25)

    label_map: Dict[int, str] = {}
    for cid, row in stats.iterrows():
        vol = row["total_qty"]
        trend = row["trend"]
        recent = row["recent"]
        cv = row["cv"]
        seasonal = row["seasonal"]

        if vol >= q75:
            tier = "High Volume"
        elif vol >= q25:
            tier = "Medium Volume"
        else:
            tier = "Low Volume"

        if recent > 15:
            momentum = "Growing"
        elif recent < -15:
            momentum = "Declining"
        elif seasonal > 2.0:
            momentum = "Seasonal"
        elif cv > 0.6:
            momentum = "Volatile"
        else:
            momentum = "Stable"

        label_map[int(cid)] = f"{tier} – {momentum}"

    return label_map


# ---------------------------------------------------------------------------
# Build cluster result DataFrame
# ---------------------------------------------------------------------------

def build_cluster_result(
    features: pd.DataFrame,
    labels: np.ndarray,
    algorithm: str,
    abc: pd.Series | None = None,
    xyz: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Combine SKU features, cluster labels, ABC, XYZ into a single result table.
    """
    result = features.copy()
    result["cluster_id"] = labels
    human_labels = label_clusters(features, labels)
    result["cluster_label"] = result["cluster_id"].map(human_labels)
    result["algorithm"] = algorithm

    if abc is not None:
        result["abc"] = abc
    if xyz is not None:
        result["xyz"] = xyz

    if abc is not None and xyz is not None:
        result["abc_xyz"] = result["abc"].astype(str) + result["xyz"].astype(str)

    return result.reset_index()
