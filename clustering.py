"""
clustering.py

Feature preparation (log transform + scaling), K-Means clustering, PCA
for visualization, cluster profiling, and turning raw cluster numbers
into human-readable RFM segment names (e.g. "Champions", "At Risk").
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURES = ["Recency", "Frequency", "Monetary", "Return_Rate"]

# Ordered best-to-worst; clusters beyond this list just get "Segment {id}".
_SEGMENT_LABELS_BY_RANK = [
    "Champions",
    "Loyal Customers",
    "Potential Loyalists",
    "At Risk",
    "Hibernating",
    "Lost",
]


def prepare_features(rfm_behavior: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Log-transform the skewed features, then standard-scale them.

    Returns the scaled features and the fitted scaler, so the same
    transformation can later be applied to a single new customer.
    """
    model_df = rfm_behavior.copy()
    for col in FEATURES:
        model_df[col] = np.log1p(model_df[col])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(model_df[FEATURES])
    scaled_df = pd.DataFrame(scaled, columns=FEATURES, index=model_df.index)

    return scaled_df, scaler


def find_best_k(scaled_df: pd.DataFrame, k_range=range(2, 11)) -> dict:
    """
    Compute inertia and silhouette score for each K in k_range, so you
    (or the dashboard's elbow chart) can pick a defensible K instead of
    guessing.
    """
    inertia, silhouette = [], []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_df)
        inertia.append(float(kmeans.inertia_))
        silhouette.append(float(silhouette_score(scaled_df, labels)))

    return {"k_range": list(k_range), "inertia": inertia, "silhouette": silhouette}


def run_kmeans(scaled_df: pd.DataFrame, k: int) -> Tuple[np.ndarray, KMeans]:
    """Fit K-Means with a chosen K and return the cluster labels + fitted model."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_df)
    return labels, kmeans


def run_pca(scaled_df: pd.DataFrame, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
    """Reduce dimensions for visualization (e.g. a 2D scatter plot of clusters)."""
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(scaled_df)
    return coords, pca


def segment_customers(rfm_behavior: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """
    End-to-end: prepare features -> run KMeans -> attach cluster labels
    to the original (unscaled) RFM + Return_Rate table.
    """
    scaled_df, _ = prepare_features(rfm_behavior)
    labels, _ = run_kmeans(scaled_df, k)

    result = rfm_behavior.copy()
    result["Cluster"] = labels
    return result


def cluster_summary(segmented: pd.DataFrame) -> pd.DataFrame:
    """Average RFM + Return_Rate per cluster — used for profiling/labeling segments."""
    return (
        segmented.groupby("Cluster")
        .agg(
            Customers=("Customer ID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Avg_Return_Rate=("Return_Rate", "mean"),
        )
        .round(2)
    )


def label_segments(summary: pd.DataFrame) -> Dict[int, str]:
    """
    Turn cluster numbers into human-readable RFM segment names.

    Clusters are ranked, not thresholded against fixed numbers, because
    "high frequency" means something different in every dataset. A
    cluster earns a better rank the lower its average Recency and the
    higher its average Frequency and Monetary value are. A cluster whose
    average Return_Rate is unusually high (more than 1 std dev above the
    mean across clusters) gets a "(High Returns)" suffix regardless of
    its value rank, since that's operationally important on its own.

    Args:
        summary: the output of cluster_summary() — indexed by Cluster.

    Returns:
        {cluster_id: "Champions"} style mapping, safe to look up any
        cluster id produced by the same KMeans run.
    """
    if summary.empty:
        return {}

    scored = summary.copy()
    scored["value_rank"] = (
        (-scored["Avg_Recency"]).rank()
        + scored["Avg_Frequency"].rank()
        + scored["Avg_Monetary"].rank()
    )
    ranked_cluster_ids = scored.sort_values("value_rank", ascending=False).index.tolist()

    names: Dict[int, str] = {}
    for i, cluster_id in enumerate(ranked_cluster_ids):
        base_name = (
            _SEGMENT_LABELS_BY_RANK[i]
            if i < len(_SEGMENT_LABELS_BY_RANK)
            else f"Segment {cluster_id}"
        )
        names[int(cluster_id)] = base_name

    return_std = scored["Avg_Return_Rate"].std()
    if return_std and return_std > 0:
        return_threshold = scored["Avg_Return_Rate"].mean() + return_std
        for cluster_id in ranked_cluster_ids:
            if scored.loc[cluster_id, "Avg_Return_Rate"] >= return_threshold:
                names[int(cluster_id)] += " (High Returns)"

    return names
