from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from feature_config import OBSERVATIONAL_CONTROL_COLUMNS


IDENTIFIER_COLUMNS = ["pl_name", "hostname"]


def safe_log10(s: pd.Series) -> pd.Series:
    """
    Apply log10 to positive values.
    Non-positive values become NaN.
    """

    values = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return np.log10(values.where(values > 0))


def select_existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """
    Return only columns that exist in df.
    """

    return [column for column in columns if column in df.columns]


def _feature_source_column(df: pd.DataFrame, feature: str) -> str | None:
    original_column = f"original_{feature}"
    if original_column in df.columns:
        return original_column
    if feature in df.columns:
        return feature
    return None


def _preserved_context_columns(df: pd.DataFrame, features: list[str]) -> list[str]:
    columns = [column for column in [*IDENTIFIER_COLUMNS, *OBSERVATIONAL_CONTROL_COLUMNS] if column in df.columns]
    for feature in features:
        for suffix in ("_was_missing", "_source"):
            candidate = f"{feature}{suffix}"
            if candidate in df.columns and candidate not in columns:
                columns.append(candidate)
    return columns


def _coerce_feature_matrix(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: dict[str, pd.Series] = {}
    source_columns: dict[str, str] = {}
    for feature in features:
        source_column = _feature_source_column(df, feature)
        if source_column is None:
            continue
        values = pd.to_numeric(df[source_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        rows[feature] = values
        source_columns[feature] = source_column
    if not rows:
        raise ValueError("Ninguna de las features solicitadas existe en el DataFrame de entrada.")
    matrix = pd.DataFrame(rows, index=df.index)
    return matrix, source_columns


def preprocess_mapper_features(
    df: pd.DataFrame,
    features: list[str],
    log10_features: list[str],
    complete_case_only: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, object, list[str]]:
    """
    Returns:
    - work_df: dataframe aligned to rows used by Mapper.
    - Z: transformed + RobustScaler-scaled numeric matrix.
    - scaler: fitted RobustScaler.
    - used_features: final feature list actually used.
    """

    requested = list(dict.fromkeys(features))
    numeric, source_columns = _coerce_feature_matrix(df, requested)
    used_features = [feature for feature in requested if feature in numeric.columns]
    if not used_features:
        raise ValueError("No hay features disponibles para construir el espacio Mapper.")

    transformed = numeric.loc[:, used_features].copy()
    log_set = set(log10_features)
    log_rows: list[dict[str, Any]] = []
    log_columns: dict[str, pd.Series] = {}

    for feature in used_features:
        values = transformed[feature]
        missing_before = int(values.isna().sum())
        nonpositive_count = 0
        if feature in log_set:
            nonpositive_count = int((values.notna() & (values <= 0)).sum())
            values = safe_log10(values)
            log_columns[f"log10_{feature}"] = values
        transformed[feature] = values
        log_rows.append(
            {
                "feature": feature,
                "source_column": source_columns[feature],
                "log10_applied": feature in log_set,
                "missing_before_transform": missing_before,
                "nonpositive_set_missing": nonpositive_count,
                "missing_after_transform": int(values.isna().sum()),
            }
        )

    valid_mask = transformed.notna().all(axis=1)
    if complete_case_only:
        row_mask = valid_mask
    else:
        row_mask = valid_mask

    if not bool(row_mask.any()):
        raise ValueError("No quedaron filas validas para Mapper despues del preprocesamiento.")

    context_columns = _preserved_context_columns(df, used_features)
    work_df = df.loc[row_mask, context_columns].copy()
    for feature in used_features:
        work_df[feature] = numeric.loc[row_mask, feature]
    for log_column, values in log_columns.items():
        work_df[log_column] = values.loc[row_mask]

    transformed_used = transformed.loc[row_mask, used_features]
    scaler = RobustScaler()
    scaled = scaler.fit_transform(transformed_used)
    Z = np.asarray(scaled, dtype=float)

    work_df.attrs["preprocessing"] = {
        "rows_before": int(len(df)),
        "rows_after": int(len(work_df)),
        "rows_dropped": int((~row_mask).sum()),
        "complete_case_only": bool(complete_case_only),
        "used_features": used_features,
        "source_columns": source_columns,
        "log10_features_used": [feature for feature in used_features if feature in log_set],
        "log_transform_audit": log_rows,
    }
    return work_df, Z, scaler, used_features
