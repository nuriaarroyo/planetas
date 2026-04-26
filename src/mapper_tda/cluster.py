from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def estimate_dbscan_eps(
    Z: np.ndarray,
    min_samples: int = 4,
    percentile: float = 90,
) -> float:
    """
    Estimate DBSCAN eps from distance to min_samples-th nearest neighbor.
    """

    if Z.ndim != 2 or Z.shape[0] == 0:
        raise ValueError("No se puede estimar eps sobre una matriz vacia.")
    if Z.shape[0] == 1:
        return 0.5

    n_neighbors = min(max(2, min_samples), Z.shape[0])
    distances, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(Z).kneighbors(Z)
    kth_distances = distances[:, -1]
    positive = kth_distances[kth_distances > 0]

    if positive.size:
        for candidate_percentile in (percentile, 95, 90, 75, 50):
            eps = float(np.percentile(positive, candidate_percentile))
            if np.isfinite(eps) and eps > 0:
                return eps

    spread = np.nanmedian(np.nanstd(Z, axis=0))
    if np.isfinite(spread) and spread > 0:
        return float(spread)
    return 1e-3


def make_clusterer(
    clusterer: str,
    Z: np.ndarray,
    min_samples: int,
    eps_percentile: float,
) -> object:
    clusterer_name = clusterer.lower()
    if clusterer_name != "dbscan":
        raise ValueError(f"Clusterer no soportado: {clusterer}. Por ahora solo se soporta 'dbscan'.")

    eps = estimate_dbscan_eps(Z, min_samples=min_samples, percentile=eps_percentile)
    return DBSCAN(eps=eps, min_samples=min_samples)
