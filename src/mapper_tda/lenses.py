from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .preprocessing import safe_log10


EPS = 1e-8


def _pca_projection(Z: np.ndarray, n_components: int) -> tuple[np.ndarray, PCA, int]:
    if Z.ndim != 2 or Z.shape[0] == 0 or Z.shape[1] == 0:
        raise ValueError("La matriz Z para construir el lens esta vacia.")
    fitted_components = min(n_components, Z.shape[0], Z.shape[1])
    if fitted_components <= 0:
        raise ValueError("No hay suficientes observaciones para PCA.")
    pca = PCA(n_components=fitted_components)
    scores = pca.fit_transform(Z)
    return scores, pca, fitted_components


def make_lens_pca2(Z: np.ndarray, random_state: int = 42) -> tuple[np.ndarray, dict]:
    """
    Return:
    - lens matrix with shape (n_samples, 2): [PC1, PC2]
    - metadata with explained variance ratio
    """

    del random_state
    scores, pca, fitted_components = _pca_projection(Z, n_components=2)
    if fitted_components == 1:
        scores = np.column_stack([scores[:, 0], np.zeros(Z.shape[0])])
    metadata = {
        "lens": "pca2",
        "explained_variance_ratio": [
            float(value) for value in np.pad(pca.explained_variance_ratio_, (0, max(0, 2 - fitted_components)))
        ],
        "n_components_fitted": int(fitted_components),
    }
    return np.asarray(scores[:, :2], dtype=float), metadata


def make_lens_density(
    Z: np.ndarray,
    k_density: int = 15,
    random_state: int = 42,
) -> tuple[np.ndarray, dict]:
    """
    Return:
    - lens matrix with shape (n_samples, 2): [PC1, log(distance_to_kth_neighbor + eps)]
    - metadata with k_density and density summary
    """

    del random_state
    scores, pca, _ = _pca_projection(Z, n_components=1)
    pc1 = scores[:, 0]

    if Z.shape[0] <= 1:
        kth_distance = np.zeros(Z.shape[0], dtype=float)
        effective_k = 0
    else:
        effective_k = min(max(1, k_density), Z.shape[0] - 1)
        neighbors = NearestNeighbors(n_neighbors=effective_k + 1)
        distances, _ = neighbors.fit(Z).kneighbors(Z)
        kth_distance = distances[:, -1]

    density_score = np.log(kth_distance + EPS)
    lens = np.column_stack([pc1, density_score])
    metadata = {
        "lens": "density",
        "explained_variance_ratio_pc1": float(pca.explained_variance_ratio_[0]),
        "k_density_requested": int(k_density),
        "k_density_effective": int(effective_k),
        "distance_summary": {
            "min": float(np.min(kth_distance)) if len(kth_distance) else 0.0,
            "median": float(np.median(kth_distance)) if len(kth_distance) else 0.0,
            "max": float(np.max(kth_distance)) if len(kth_distance) else 0.0,
        },
        "density_score_summary": {
            "min": float(np.min(density_score)) if len(density_score) else 0.0,
            "median": float(np.median(density_score)) if len(density_score) else 0.0,
            "max": float(np.max(density_score)) if len(density_score) else 0.0,
        },
    }
    return np.asarray(lens, dtype=float), metadata


def _domain_columns_for_space(space: str) -> list[str]:
    if space == "phys":
        return ["pl_rade", "pl_dens"]
    if space == "orb":
        return ["pl_orbper", "pl_insol"]
    if space == "joint":
        return ["pl_rade", "pl_orbper"]
    raise ValueError(f"Espacio no soportado para lens domain: {space}")


def make_lens_domain(
    work_df: pd.DataFrame,
    space: str,
) -> tuple[np.ndarray, dict]:
    columns = _domain_columns_for_space(space)
    domain_values: list[np.ndarray] = []
    used_columns: list[str] = []

    for feature in columns:
        log_column = f"log10_{feature}"
        if log_column in work_df.columns:
            values = pd.to_numeric(work_df[log_column], errors="coerce")
            used_columns.append(log_column)
        elif feature in work_df.columns:
            values = safe_log10(work_df[feature])
            used_columns.append(log_column)
        else:
            raise ValueError(
                f"No se pudo construir el lens domain para '{space}'. "
                f"Falta la columna '{feature}' o '{log_column}'."
            )
        if values.isna().any():
            raise ValueError(
                f"El lens domain para '{space}' requiere valores positivos y completos en '{feature}'."
            )
        domain_values.append(values.to_numpy(dtype=float))

    lens = np.column_stack(domain_values)
    metadata = {
        "lens": "domain",
        "space": space,
        "columns": used_columns,
    }
    return lens, metadata
