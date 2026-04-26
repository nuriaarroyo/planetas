from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from feature_config import (
    IDENTIFIER_COLUMNS,
    IMPUTATION_VALUE_BOUNDS,
    LOG10_FEATURES,
    MAPPER_JOINT_FEATURES,
    MAPPER_ORB_FEATURES,
    MAPPER_ORB_OPTIONAL_FEATURES,
    MAPPER_PHYS_FEATURES,
    NON_MODEL_SUBSTRINGS,
    OBSERVATIONAL_CONTROL_COLUMNS,
    STELLAR_CONTEXT_FEATURES,
)
from imputation.io import write_json
from imputation.steps.baseline_imputers import impute_with_iterative, impute_with_median
from imputation.steps.constraints import apply_feature_bounds
from imputation.steps.knn_imputer import impute_with_knn
from imputation.steps.log_transform import apply_log10_transform, invert_log10_transform, log_feature_subset
from imputation.steps.physical_derivation import PhysicalDerivationAudit, apply_physical_derivations
from imputation.steps.scaling import invert_robust_scale, robust_scale


VALID_METHODS = ("median", "knn", "iterative", "compare")
BASE_METHODS = ("median", "knn", "iterative")
EPS = 1e-12
REPORT_FEATURES = ["pl_rade", "pl_bmasse", "pl_dens", "pl_orbper", "pl_orbsmax", "pl_insol", "pl_eqt"]
POSITIVE_LOG_FEATURES = {"pl_rade", "pl_bmasse", "pl_dens", "pl_orbper", "pl_orbsmax", "pl_insol"}
SCATTER_SPECS = [
    ("pl_bmasse", "pl_rade", "scatter_mass_radius", "Planet mass vs radius"),
    ("pl_dens", "pl_rade", "scatter_density_radius", "Planet density vs radius"),
    ("pl_orbper", "pl_orbsmax", "scatter_orbper_orbsmax", "Orbital period vs semimajor axis"),
    ("pl_insol", "pl_eqt", "scatter_insol_eqt", "Insolation vs equilibrium temperature"),
]


@dataclass(frozen=True)
class ImputationConfig:
    method: str = "iterative"
    visualized_method: str = "iterative"
    n_neighbors: int = 15
    weights: str = "distance"
    max_missing_pct: float = 60.0
    validation_mask_frac: float = 0.15
    random_state: int = 42
    n_multiple_imputations: int = 1
    include_stellar_context: bool = False
    include_orbital_eccentricity: bool = False
    iterative_max_iter: int = 20
    value_bounds: Mapping[str, tuple[float | None, float | None]] = field(
        default_factory=lambda: dict(IMPUTATION_VALUE_BOUNDS)
    )


@dataclass
class MethodResult:
    method: str
    seed: int
    full_df: pd.DataFrame
    original_units: pd.DataFrame
    mapper_scaled: pd.DataFrame
    transformed_before: pd.DataFrame
    imputed_values_long: pd.DataFrame
    missingness_after: pd.DataFrame
    feature_coverage: pd.DataFrame
    validation_metrics: pd.DataFrame
    validation_predictions: pd.DataFrame
    mapper_tables: dict[str, pd.DataFrame]
    summary: dict[str, Any]


@dataclass
class ImputationResult:
    raw_df: pd.DataFrame
    prepared_df: pd.DataFrame
    features_requested: list[str]
    features_included: list[str]
    excluded_features: pd.DataFrame
    missingness_before: pd.DataFrame
    log_transform_audit: pd.DataFrame
    physical_audit: PhysicalDerivationAudit
    methods: dict[str, MethodResult]
    method_comparison: pd.DataFrame


def default_log_features(features: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(log_feature_subset(features, LOG10_FEATURES))


def methods_to_run(config: ImputationConfig) -> list[str]:
    if config.method == "compare":
        return list(BASE_METHODS)
    if config.method not in BASE_METHODS:
        raise ValueError(f"Metodo no soportado: {config.method}")
    return [config.method]


def select_visualized_key(result: ImputationResult, visualized_method: str) -> str:
    if visualized_method in result.methods:
        return visualized_method
    if result.methods:
        for key, method_result in result.methods.items():
            if method_result.method == visualized_method:
                return key
    if result.methods:
        return next(iter(result.methods))
    raise ValueError("No hay resultados de imputacion para visualizar.")


def is_non_model_column(column: str) -> bool:
    name = column.lower()
    return any(token in name for token in NON_MODEL_SUBSTRINGS)


def requested_mapper_features(config: ImputationConfig) -> list[str]:
    features = list(MAPPER_JOINT_FEATURES)
    if config.include_orbital_eccentricity:
        features.extend(MAPPER_ORB_OPTIONAL_FEATURES)
    if config.include_stellar_context:
        features.extend(STELLAR_CONTEXT_FEATURES)
    return list(dict.fromkeys(features))


def coerce_numeric_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    matrix = df.loc[:, features].copy()
    for column in matrix.columns:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return matrix


def _source_category(source: object) -> str:
    value = "" if pd.isna(source) else str(source)
    if value == "observed":
        return "observed"
    if value in {"derived_density", "derived_kepler"}:
        return "physically_derived"
    if value.startswith("imputed_"):
        return "imputed"
    if value == "excluded_too_missing":
        return "excluded_too_missing"
    return "missing"


def _set_source_flags(df: pd.DataFrame, feature: str) -> None:
    source_col = f"{feature}_source"
    if source_col not in df.columns:
        source = pd.Series(pd.NA, index=df.index, dtype="object")
    else:
        source = df[source_col]
    categories = source.map(_source_category)
    df[f"{feature}_was_observed"] = categories.eq("observed").to_numpy()
    df[f"{feature}_was_physically_derived"] = categories.eq("physically_derived").to_numpy()
    df[f"{feature}_was_imputed"] = categories.eq("imputed").to_numpy()


def initialize_sources_and_flags(
    raw_df: pd.DataFrame,
    prepared_df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    out = prepared_df.copy()
    for feature in features:
        if feature not in out.columns:
            continue
        source_col = f"{feature}_source"
        if source_col not in out.columns:
            source = pd.Series(pd.NA, index=out.index, dtype="object")
            source.loc[pd.to_numeric(out[feature], errors="coerce").notna()] = "observed"
            out[source_col] = source
        raw_missing = raw_df[feature].isna() if feature in raw_df.columns else pd.Series(True, index=out.index)
        out[f"{feature}_was_missing"] = raw_missing.astype(bool).to_numpy()
        _set_source_flags(out, feature)
    return out


def select_imputation_features(
    prepared_df: pd.DataFrame,
    requested_features: list[str],
    max_missing_pct: float,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available = [feature for feature in requested_features if feature in prepared_df.columns]
    rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    if not available:
        raise ValueError("Ninguna feature solicitada existe en el DataFrame.")

    numeric = coerce_numeric_features(prepared_df, available)
    transformed, log_audit = apply_log10_transform(numeric, default_log_features(available))
    selected: list[str] = []
    for feature in requested_features:
        if feature not in prepared_df.columns:
            excluded_rows.append(
                {
                    "feature": feature,
                    "reason": "missing_column",
                    "missing_pct": 100.0,
                }
            )
            continue
        if is_non_model_column(feature) or feature in IDENTIFIER_COLUMNS:
            excluded_rows.append(
                {
                    "feature": feature,
                    "reason": "non_model_column",
                    "missing_pct": float(prepared_df[feature].isna().mean() * 100),
                }
            )
            continue
        missing_pct = float(transformed[feature].isna().mean() * 100)
        if missing_pct > max_missing_pct:
            excluded_rows.append(
                {
                    "feature": feature,
                    "reason": "excluded_too_missing",
                    "missing_pct": round(missing_pct, 4),
                }
            )
            continue
        selected.append(feature)
        rows.append(
            {
                "feature": feature,
                "missing_pct_for_model": round(missing_pct, 4),
                "log10_applied": feature in LOG10_FEATURES,
            }
        )

    if not selected:
        raise ValueError("Todas las features quedaron excluidas por nulos o reglas de modelado.")

    selected_profile = pd.DataFrame(rows)
    excluded = pd.DataFrame(excluded_rows, columns=["feature", "reason", "missing_pct"])
    return selected, excluded, transformed[selected], log_audit.to_frame()


def build_missingness_profile_before(
    raw_df: pd.DataFrame,
    prepared_df: pd.DataFrame,
    features: list[str],
    excluded_features: pd.DataFrame,
) -> pd.DataFrame:
    excluded_lookup = (
        excluded_features.set_index("feature")["reason"].to_dict() if not excluded_features.empty else {}
    )
    rows: list[dict[str, Any]] = []
    total = len(prepared_df)
    for feature in features:
        raw_missing = int(raw_df[feature].isna().sum()) if feature in raw_df.columns else total
        prepared_missing = int(prepared_df[feature].isna().sum()) if feature in prepared_df.columns else total
        rows.append(
            {
                "feature": feature,
                "rows": total,
                "missing_raw": raw_missing,
                "missing_raw_pct": round(raw_missing / total * 100, 4) if total else np.nan,
                "missing_after_physical_derivation": prepared_missing,
                "missing_after_physical_derivation_pct": round(prepared_missing / total * 100, 4)
                if total
                else np.nan,
                "excluded_reason": excluded_lookup.get(feature, ""),
            }
        )
    return pd.DataFrame(rows)


def _impute_scaled_matrix(method: str, scaled: pd.DataFrame, config: ImputationConfig, seed: int) -> pd.DataFrame:
    if method == "knn":
        # KNN preserves local neighborhoods in the observed feature space, which
        # is the relevant geometry for Mapper/TDA.
        return impute_with_knn(scaled, n_neighbors=config.n_neighbors, weights=config.weights)
    if method == "median":
        return impute_with_median(scaled)
    if method == "iterative":
        return impute_with_iterative(scaled, random_state=seed, max_iter=config.iterative_max_iter)
    raise ValueError(f"Metodo no soportado: {method}")


def impute_numeric_matrix(
    numeric: pd.DataFrame,
    method: str,
    config: ImputationConfig,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log_features = default_log_features(list(numeric.columns))
    transformed, log_audit = apply_log10_transform(numeric, log_features)
    scaled, scaler = robust_scale(transformed)
    scaled_imputed = _impute_scaled_matrix(method, scaled, config, seed)
    transformed_imputed = invert_robust_scale(scaled_imputed, scaler)
    original_units = invert_log10_transform(transformed_imputed, log_features)
    original_units, _ = apply_feature_bounds(original_units, config.value_bounds)
    for feature in set(original_units.columns) & POSITIVE_LOG_FEATURES:
        original_units[feature] = pd.to_numeric(original_units[feature], errors="coerce").clip(lower=EPS)

    transformed_clipped, _ = apply_log10_transform(original_units, log_features)
    mapper_scaled = pd.DataFrame(
        scaler.transform(transformed_clipped),
        index=numeric.index,
        columns=numeric.columns,
    )
    return original_units, mapper_scaled, transformed, scaled, log_audit.to_frame()


def apply_imputation_to_full_frame(
    raw_df: pd.DataFrame,
    prepared_df: pd.DataFrame,
    features: list[str],
    original_units: pd.DataFrame,
    transformed_before: pd.DataFrame,
    excluded_features: pd.DataFrame,
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = prepared_df.copy()
    missing_for_model = transformed_before.isna()
    excluded = set(excluded_features["feature"]) if not excluded_features.empty else set()

    for feature in features:
        out[feature] = original_units[feature]
        source_col = f"{feature}_source"
        if source_col not in out.columns:
            source = pd.Series(pd.NA, index=out.index, dtype="object")
            source.loc[pd.to_numeric(prepared_df[feature], errors="coerce").notna()] = "observed"
        else:
            source = out[source_col].copy()
            source = source.where(source.notna(), pd.NA)
        source.loc[missing_for_model[feature]] = f"imputed_{method}"
        out[source_col] = source
        raw_missing = raw_df[feature].isna() if feature in raw_df.columns else pd.Series(True, index=out.index)
        out[f"{feature}_was_missing"] = raw_missing.astype(bool).to_numpy()
        _set_source_flags(out, feature)

    for feature in excluded:
        if feature not in out.columns:
            continue
        source_col = f"{feature}_source"
        out[source_col] = "excluded_too_missing"
        raw_missing = raw_df[feature].isna() if feature in raw_df.columns else pd.Series(True, index=out.index)
        out[f"{feature}_was_missing"] = raw_missing.astype(bool).to_numpy()
        _set_source_flags(out, feature)

    long_rows: list[dict[str, Any]] = []
    id_cols = [column for column in ["pl_name", "hostname"] if column in out.columns]
    for feature in features:
        mask = missing_for_model[feature]
        if not mask.any():
            continue
        values = out.loc[mask, [feature, f"{feature}_source", f"{feature}_was_missing", *id_cols]].copy()
        for index, row in values.iterrows():
            payload = {
                "row_index": index,
                "feature": feature,
                "method": method,
                "imputed_value": row[feature],
                "source": row[f"{feature}_source"],
                "source_category": _source_category(row[f"{feature}_source"]),
                "was_missing": bool(row[f"{feature}_was_missing"]),
            }
            for column in id_cols:
                payload[column] = row[column]
            long_rows.append(payload)

    out["imputation_method"] = method
    return out, pd.DataFrame(long_rows)


def missingness_profile_after(
    full_df: pd.DataFrame,
    features: list[str],
    excluded_features: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    total = len(full_df)
    excluded = set(excluded_features["feature"]) if not excluded_features.empty else set()
    rows: list[dict[str, Any]] = []
    for feature in list(dict.fromkeys([*features, *excluded])):
        if feature not in full_df.columns:
            continue
        source_col = f"{feature}_source"
        source = full_df[source_col] if source_col in full_df.columns else pd.Series(pd.NA, index=full_df.index)
        source_categories = source.map(_source_category)
        missing = int(full_df[feature].isna().sum())
        rows.append(
            {
                "feature": feature,
                "rows": total,
                "method": method,
                "missing_after": missing,
                "missing_after_pct": round(missing / total * 100, 4) if total else np.nan,
                "observed_count": int((source == "observed").sum()),
                "derived_density_count": int((source == "derived_density").sum()),
                "derived_kepler_count": int((source == "derived_kepler").sum()),
                "physically_derived_count": int(source_categories.eq("physically_derived").sum()),
                "imputed_count": int(source.astype("string").str.startswith("imputed_", na=False).sum()),
                "excluded_too_missing_count": int((source == "excluded_too_missing").sum()),
                "still_missing_count": missing,
            }
        )
    return pd.DataFrame(rows)


def validate_imputed_feature_matrix(full_df: pd.DataFrame, features: list[str]) -> None:
    matrix = coerce_numeric_features(full_df, features)
    if matrix.isna().any().any():
        bad = matrix.columns[matrix.isna().any()].tolist()
        raise ValueError(f"La matriz imputada aun contiene NaN en: {bad}")
    if not np.isfinite(matrix.to_numpy()).all():
        bad = matrix.columns[~np.isfinite(matrix.to_numpy()).all(axis=0)].tolist()
        raise ValueError(f"La matriz imputada contiene valores no finitos en: {bad}")
    non_positive = [
        feature
        for feature in features
        if feature in POSITIVE_LOG_FEATURES and (pd.to_numeric(full_df[feature], errors="coerce") <= 0).any()
    ]
    if non_positive:
        raise ValueError(f"Features positivas usadas en log tienen valores <= 0: {non_positive}")


def feature_groups(include_orbital_eccentricity: bool, include_stellar_context: bool) -> dict[str, list[str]]:
    orb = list(MAPPER_ORB_FEATURES)
    joint = list(MAPPER_JOINT_FEATURES)
    joint_optional = list(MAPPER_JOINT_FEATURES) + list(MAPPER_ORB_OPTIONAL_FEATURES)
    if include_orbital_eccentricity:
        orb += list(MAPPER_ORB_OPTIONAL_FEATURES)
        joint += list(MAPPER_ORB_OPTIONAL_FEATURES)
    if include_stellar_context:
        joint += list(STELLAR_CONTEXT_FEATURES)
        joint_optional += list(STELLAR_CONTEXT_FEATURES)
    return {
        "MAPPER_PHYS_FEATURES": list(MAPPER_PHYS_FEATURES),
        "MAPPER_ORB_FEATURES": orb,
        "MAPPER_JOINT_FEATURES": joint,
        "MAPPER_JOINT_FEATURES + MAPPER_ORB_OPTIONAL_FEATURES": joint_optional,
    }


def coverage_table(
    prepared_df: pd.DataFrame,
    full_df: pd.DataFrame,
    included_features: list[str],
    config: ImputationConfig,
    method: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    included = set(included_features)
    for group, columns in feature_groups(
        config.include_orbital_eccentricity,
        config.include_stellar_context,
    ).items():
        available_before = [column for column in columns if column in prepared_df.columns]
        available_after = [column for column in columns if column in full_df.columns and column in included]
        missing_columns = [column for column in columns if column not in prepared_df.columns]
        excluded_columns = [column for column in columns if column in prepared_df.columns and column not in included]
        before_complete = (
            int(prepared_df[available_before].notna().all(axis=1).sum())
            if len(available_before) == len(columns)
            else 0
        )
        after_complete = (
            int(full_df[available_after].notna().all(axis=1).sum())
            if len(available_after) == len(columns)
            else 0
        )
        rows.append(
            {
                "feature_group": group,
                "method": method,
                "n_features": len(columns),
                "before_complete_rows": before_complete,
                "before_complete_pct": round(before_complete / len(prepared_df) * 100, 4)
                if len(prepared_df)
                else np.nan,
                "after_complete_rows": after_complete,
                "after_complete_pct": round(after_complete / len(full_df) * 100, 4) if len(full_df) else np.nan,
                "missing_columns": ", ".join(missing_columns),
                "excluded_columns": ", ".join(excluded_columns),
                "columns": ", ".join(columns),
            }
        )
    return pd.DataFrame(rows)


def _mape_or_nan(truth: pd.Series, prediction: pd.Series) -> float:
    if truth.empty or not (truth > EPS).all():
        return np.nan
    return float(((prediction - truth).abs() / truth).mean() * 100)


def validation_metric_row(
    method: str,
    feature: str,
    truth: pd.Series,
    prediction: pd.Series,
    validation_sources: pd.Series,
) -> dict[str, Any]:
    errors = prediction - truth
    abs_errors = errors.abs()
    positive = (truth > EPS) & (prediction > EPS)
    log_errors = np.log10(prediction[positive]) - np.log10(truth[positive])
    source_categories = validation_sources.map(_source_category)
    n_observed = int(source_categories.eq("observed").sum())
    n_derived = int(source_categories.eq("physically_derived").sum())
    if n_observed and n_derived:
        validation_basis = "observed_and_physically_derived"
    elif n_derived:
        validation_basis = "physically_derived"
    else:
        validation_basis = "observed"
    return {
        "method": method,
        "feature": feature,
        "n_validated": int(len(truth)),
        "n_validated_observed_original": n_observed,
        "n_validated_physically_derived": n_derived,
        "validation_basis": validation_basis,
        "mae": float(abs_errors.mean()) if len(abs_errors) else np.nan,
        "rmse": float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else np.nan,
        "medae": float(abs_errors.median()) if len(abs_errors) else np.nan,
        "mape": _mape_or_nan(truth, prediction),
        "log_mae": float(log_errors.abs().mean()) if len(log_errors) else np.nan,
        "log_rmse": float(np.sqrt(np.mean(np.square(log_errors)))) if len(log_errors) else np.nan,
        "spearman": float(truth.corr(prediction, method="spearman"))
        if len(truth) >= 2 and truth.nunique() > 1 and prediction.nunique() > 1
        else np.nan,
        "pearson": float(truth.corr(prediction, method="pearson"))
        if len(truth) >= 2 and truth.nunique() > 1 and prediction.nunique() > 1
        else np.nan,
    }


def build_validation_masks(
    numeric: pd.DataFrame,
    config: ImputationConfig,
) -> dict[str, pd.Index]:
    transformed, _ = apply_log10_transform(numeric, default_log_features(list(numeric.columns)))
    rng = np.random.default_rng(config.random_state)
    mask_by_feature: dict[str, pd.Index] = {}
    for feature in numeric.columns:
        valid_idx = transformed.index[transformed[feature].notna()]
        if len(valid_idx) < 2 or config.validation_mask_frac <= 0:
            mask_by_feature[feature] = pd.Index([])
            continue
        n_mask = max(1, int(round(len(valid_idx) * config.validation_mask_frac)))
        n_mask = min(n_mask, len(valid_idx) - 1)
        chosen = rng.choice(valid_idx.to_numpy(), size=n_mask, replace=False)
        mask_by_feature[feature] = pd.Index(chosen)
    return mask_by_feature


def run_masked_validation(
    numeric: pd.DataFrame,
    prepared_df: pd.DataFrame,
    method: str,
    config: ImputationConfig,
    seed: int,
    mask_by_feature: dict[str, pd.Index],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    masked = numeric.copy()
    for feature, chosen in mask_by_feature.items():
        if len(chosen) == 0:
            continue
        masked.loc[chosen, feature] = np.nan

    imputed, _, _, _, _ = impute_numeric_matrix(masked, method, config, seed)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for feature, idx in mask_by_feature.items():
        if len(idx) == 0:
            continue
        truth = numeric.loc[idx, feature]
        prediction = imputed.loc[idx, feature]
        source_col = f"{feature}_source"
        if source_col in prepared_df.columns:
            validation_sources = prepared_df.loc[idx, source_col]
        else:
            validation_sources = pd.Series("observed", index=idx, dtype="object")
        metric_rows.append(validation_metric_row(method, feature, truth, prediction, validation_sources))
        for row_index in idx:
            true_value = numeric.loc[row_index, feature]
            predicted_value = imputed.loc[row_index, feature]
            source = prepared_df.loc[row_index, source_col] if source_col in prepared_df.columns else "observed"
            prediction_rows.append(
                {
                    "method": method,
                    "feature": feature,
                    "row_index": row_index,
                    "validation_source": source,
                    "validation_source_category": _source_category(source),
                    "true_value": true_value,
                    "predicted_value": predicted_value,
                    "error": predicted_value - true_value,
                    "absolute_error": abs(predicted_value - true_value),
                }
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def scaled_mapper_table(
    df: pd.DataFrame,
    features: list[str],
    complete_only: bool,
) -> pd.DataFrame:
    available = [feature for feature in features if feature in df.columns]
    context_cols = [
        column
        for column in ["pl_name", "hostname", *OBSERVATIONAL_CONTROL_COLUMNS]
        if column in df.columns and column not in available
    ]
    if not available:
        return df.loc[:, context_cols].copy()

    values = coerce_numeric_features(df, available)
    if complete_only:
        source_ok = pd.Series(True, index=df.index)
        for feature in available:
            source_col = f"{feature}_source"
            if source_col in df.columns:
                as_string = df[source_col].astype("string")
                source_ok &= ~as_string.str.startswith("imputed_", na=False)
                source_ok &= as_string.ne("excluded_too_missing").fillna(False)
        mask = values.notna().all(axis=1) & source_ok
    else:
        mask = values.notna().all(axis=1)

    table = df.loc[mask, context_cols].copy()
    selected_values = values.loc[mask]
    if selected_values.empty:
        return table

    transformed, _ = apply_log10_transform(selected_values, default_log_features(available))
    scaled, _ = robust_scale(transformed)
    for feature in available:
        table[feature] = scaled[feature]
    for feature in available:
        table[f"original_{feature}"] = selected_values[feature]
    for feature in available:
        flag_col = f"{feature}_was_missing"
        source_col = f"{feature}_source"
        if flag_col in df.columns:
            table[flag_col] = df.loc[mask, flag_col].astype(bool)
        if source_col in df.columns:
            table[source_col] = df.loc[mask, source_col]
    return table


def build_mapper_feature_tables(
    df: pd.DataFrame,
    method: str = "knn",
    include_stellar_context: bool = False,
    include_orbital_eccentricity: bool = False,
) -> dict[str, pd.DataFrame]:
    """Build complete-case and imputed Mapper tables in scaled feature space."""

    del method
    phys = list(MAPPER_PHYS_FEATURES)
    orb = list(MAPPER_ORB_FEATURES)
    if include_orbital_eccentricity:
        orb += list(MAPPER_ORB_OPTIONAL_FEATURES)
    joint = list(dict.fromkeys([*phys, *orb]))
    if include_stellar_context:
        joint += [feature for feature in STELLAR_CONTEXT_FEATURES if feature not in joint]

    return {
        "phys_complete": scaled_mapper_table(df, phys, complete_only=True),
        "orb_complete": scaled_mapper_table(df, orb, complete_only=True),
        "joint_complete": scaled_mapper_table(df, joint, complete_only=True),
        "phys_imputed": scaled_mapper_table(df, phys, complete_only=False),
        "orb_imputed": scaled_mapper_table(df, orb, complete_only=False),
        "joint_imputed": scaled_mapper_table(df, joint, complete_only=False),
    }


def method_summary(
    method: str,
    config: ImputationConfig,
    features: list[str],
    excluded_features: pd.DataFrame,
    full_df: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    physical_audit: PhysicalDerivationAudit,
) -> dict[str, Any]:
    feature_counts = {}
    for feature in features:
        source = full_df.get(f"{feature}_source", pd.Series(dtype="object"))
        feature_counts[feature] = {
            "observed": int((source == "observed").sum()),
            "derived_density": int((source == "derived_density").sum()),
            "derived_kepler": int((source == "derived_kepler").sum()),
            "imputed": int(source.astype("string").str.startswith("imputed_", na=False).sum()),
            "still_missing": int(full_df[feature].isna().sum()),
        }
    return {
        "method": method,
        "features_imputed": features,
        "excluded_features": excluded_features.to_dict(orient="records"),
        "n_rows": int(len(full_df)),
        "n_features_fully_imputed": int(sum(full_df[feature].isna().sum() == 0 for feature in features)),
        "feature_counts": feature_counts,
        "validation": validation_metrics.to_dict(orient="records"),
        "config": {
            "n_neighbors": config.n_neighbors,
            "weights": config.weights,
            "max_missing_pct": config.max_missing_pct,
            "validation_mask_frac": config.validation_mask_frac,
            "random_state": config.random_state,
            "include_stellar_context": config.include_stellar_context,
            "include_orbital_eccentricity": config.include_orbital_eccentricity,
        },
        "physical_derivations": {
            "density": asdict(physical_audit.density),
            "kepler": asdict(physical_audit.kepler),
        },
        "warning": (
            "Los valores imputados no son observaciones astronomicas. Sirven para analisis exploratorio, "
            "Mapper/TDA y sensibilidad estadistica. Toda conclusion debe compararse contra casos completos "
            "y contra varios metodos de imputacion."
        ),
    }


def build_method_comparison(methods: dict[str, MethodResult]) -> pd.DataFrame:
    if not methods:
        return pd.DataFrame()
    metrics = pd.concat([result.validation_metrics for result in methods.values()], ignore_index=True)
    if metrics.empty:
        return pd.DataFrame(
            [
                {
                    "method": method,
                    "mean_mae_rank": np.nan,
                    "mean_rmse_rank": np.nan,
                    "n_columns_validated": 0,
                    "n_features_fully_imputed": result.summary["n_features_fully_imputed"],
                    "notes": "No validation metrics were available.",
                }
                for method, result in methods.items()
            ]
        )

    metrics["mae_rank"] = metrics.groupby("feature")["mae"].rank(method="average")
    metrics["rmse_rank"] = metrics.groupby("feature")["rmse"].rank(method="average")
    best_method = (
        metrics.groupby("method")["mae_rank"].mean().sort_values().index[0]
        if metrics["mae_rank"].notna().any()
        else ""
    )
    rows: list[dict[str, Any]] = []
    for method, result in methods.items():
        subset = metrics[metrics["method"] == method]
        notes = []
        if method == best_method:
            notes.append("Lowest validation error among tested methods.")
        if method == "knn":
            notes.append("Default for local-neighborhood Mapper; compare topology against complete cases.")
        if method == "median":
            notes.append("Robust baseline.")
        if method == "iterative":
            notes.append("Advanced sensitivity; may impose global relationships.")
        notes.append("Topological stability is not inferred here; inspect Mapper outputs and bootstraps.")
        rows.append(
            {
                "method": method,
                "mean_mae_rank": subset["mae_rank"].mean(),
                "mean_rmse_rank": subset["rmse_rank"].mean(),
                "n_columns_validated": int(subset["feature"].nunique()),
                "n_features_fully_imputed": result.summary["n_features_fully_imputed"],
                "notes": " ".join(notes),
            }
        )
    return pd.DataFrame(rows)


def run_imputation_pipeline(df: pd.DataFrame, config: ImputationConfig) -> ImputationResult:
    if config.method not in VALID_METHODS:
        raise ValueError(f"--method debe ser uno de {VALID_METHODS}")

    raw_df = df.copy(deep=True)
    prepared_df, physical_audit = apply_physical_derivations(df)
    requested_features = requested_mapper_features(config)
    prepared_df = initialize_sources_and_flags(raw_df, prepared_df, requested_features)
    features, excluded_features, transformed_before, log_transform_audit = select_imputation_features(
        prepared_df,
        requested_features,
        config.max_missing_pct,
    )
    prepared_df = initialize_sources_and_flags(raw_df, prepared_df, [*features, *excluded_features["feature"].tolist()])
    missingness_before = build_missingness_profile_before(raw_df, prepared_df, requested_features, excluded_features)
    numeric = coerce_numeric_features(prepared_df, features)
    validation_masks = build_validation_masks(numeric, config)

    results: dict[str, MethodResult] = {}
    for method in methods_to_run(config):
        seeds = [config.random_state]
        if method == "iterative" and config.n_multiple_imputations > 1:
            seeds = [config.random_state + offset for offset in range(config.n_multiple_imputations)]
        for index, seed in enumerate(seeds, start=1):
            result_key = method if index == 1 else f"{method}_{index:02d}"
            original_units, mapper_scaled, transformed, _, _ = impute_numeric_matrix(numeric, method, config, seed)
            full_df, imputed_values_long = apply_imputation_to_full_frame(
                raw_df=raw_df,
                prepared_df=prepared_df,
                features=features,
                original_units=original_units,
                transformed_before=transformed,
                excluded_features=excluded_features,
                method=method,
            )
            validate_imputed_feature_matrix(full_df, features)
            after = missingness_profile_after(full_df, features, excluded_features, method)
            coverage = coverage_table(prepared_df, full_df, features, config, method)
            validation_metrics, validation_predictions = run_masked_validation(
                numeric,
                prepared_df,
                method,
                config,
                seed,
                validation_masks,
            )
            mapper_tables = build_mapper_feature_tables(
                full_df,
                method=method,
                include_stellar_context=config.include_stellar_context,
                include_orbital_eccentricity=config.include_orbital_eccentricity,
            )
            summary = method_summary(
                method=method,
                config=config,
                features=features,
                excluded_features=excluded_features,
                full_df=full_df,
                validation_metrics=validation_metrics,
                physical_audit=physical_audit,
            )
            results[result_key] = MethodResult(
                method=method,
                seed=seed,
                full_df=full_df,
                original_units=original_units,
                mapper_scaled=mapper_scaled,
                transformed_before=transformed,
                imputed_values_long=imputed_values_long,
                missingness_after=after,
                feature_coverage=coverage,
                validation_metrics=validation_metrics,
                validation_predictions=validation_predictions,
                mapper_tables=mapper_tables,
                summary=summary,
            )

    comparison = build_method_comparison(results)
    return ImputationResult(
        raw_df=raw_df,
        prepared_df=prepared_df,
        features_requested=requested_features,
        features_included=features,
        excluded_features=excluded_features,
        missingness_before=missingness_before,
        log_transform_audit=log_transform_audit,
        physical_audit=physical_audit,
        methods=results,
        method_comparison=comparison,
    )


def _html_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "<p>No rows.</p>"
    table = df.head(max_rows).to_html(index=False, border=0, classes="data-table")
    return f"<div class='table-wrap'>{table}</div>"


def _fig_to_html(fig: go.Figure) -> str:
    fig.update_layout(
        width=1060,
        height=max(fig.layout.height or 560, 560),
        margin=dict(l=80, r=40, t=90, b=95),
        legend=dict(orientation="h", yanchor="bottom", y=-0.32, xanchor="left", x=0),
        title=dict(x=0.01, xanchor="left"),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _report_section(title: str, body: str, note: str | None = None) -> str:
    note_html = f"<p class='section-note'>{note}</p>" if note else ""
    return f"<section><h2>{title}</h2>{note_html}{body}</section>"


def _all_validation_metrics(result: ImputationResult) -> pd.DataFrame:
    frames = [method.validation_metrics for method in result.methods.values() if not method.validation_metrics.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _source_counts_long(method_result: MethodResult) -> pd.DataFrame:
    count_columns = [
        "observed_count",
        "physically_derived_count",
        "imputed_count",
        "still_missing_count",
    ]
    available = [column for column in count_columns if column in method_result.missingness_after.columns]
    if not available:
        return pd.DataFrame()
    long = method_result.missingness_after.melt(
        id_vars=["feature"],
        value_vars=available,
        var_name="source",
        value_name="count",
    )
    long["source"] = long["source"].str.replace("_count", "", regex=False)
    return long[long["count"] > 0]


def _method_comparison_fig(comparison: pd.DataFrame) -> str:
    if comparison.empty:
        return ""
    long = comparison.melt(
        id_vars=["method"],
        value_vars=["mean_mae_rank", "mean_rmse_rank"],
        var_name="rank_type",
        value_name="mean_rank",
    )
    fig = px.bar(
        long,
        x="method",
        y="mean_rank",
        color="rank_type",
        barmode="group",
        title="Method comparison: lower rank is better in masked validation",
    )
    fig.update_layout(yaxis_title="Mean rank across features", xaxis_title="Method")
    return _fig_to_html(fig)


def _missingness_fig(result: ImputationResult, method_result: MethodResult) -> str:
    before = result.missingness_before[["feature", "missing_raw_pct", "missing_after_physical_derivation_pct"]]
    after = method_result.missingness_after[["feature", "missing_after_pct"]]
    plot_df = before.merge(after, on="feature", how="outer")
    plot_df = plot_df.melt(id_vars=["feature"], var_name="stage", value_name="missing_pct")
    plot_df["stage"] = plot_df["stage"].map(
        {
            "missing_raw_pct": "raw CSV",
            "missing_after_physical_derivation_pct": "after physical derivation",
            "missing_after_pct": f"after {method_result.method} imputation",
        }
    )
    fig = px.bar(
        plot_df,
        x="feature",
        y="missing_pct",
        color="stage",
        barmode="group",
        title="Missingness reduction by feature",
    )
    fig.update_layout(yaxis_title="% missing", xaxis_title="Feature", xaxis_tickangle=-25)
    return _fig_to_html(fig)


def _source_composition_fig(method_result: MethodResult) -> str:
    source_long = _source_counts_long(method_result)
    if source_long.empty:
        return ""
    fig = px.bar(
        source_long,
        x="feature",
        y="count",
        color="source",
        barmode="stack",
        title="Where final values came from",
    )
    fig.update_layout(yaxis_title="Rows", xaxis_title="Feature", xaxis_tickangle=-25)
    return _fig_to_html(fig)


def _coverage_fig(method_result: MethodResult) -> str:
    coverage = method_result.feature_coverage.copy()
    if coverage.empty:
        return ""
    long = coverage.melt(
        id_vars=["feature_group"],
        value_vars=["before_complete_pct", "after_complete_pct"],
        var_name="stage",
        value_name="complete_pct",
    )
    long["stage"] = long["stage"].map(
        {
            "before_complete_pct": "before imputation",
            "after_complete_pct": "after imputation",
        }
    )
    fig = px.bar(
        long,
        x="feature_group",
        y="complete_pct",
        color="stage",
        barmode="group",
        title="Complete-row coverage for Mapper feature groups",
    )
    fig.update_layout(yaxis_title="% complete rows", xaxis_title="", xaxis_tickangle=-15)
    return _fig_to_html(fig)


def _validation_figures(result: ImputationResult, method_result: MethodResult) -> str:
    metrics = _all_validation_metrics(result)
    snippets: list[str] = []
    if not metrics.empty:
        fig = px.bar(
            metrics,
            x="feature",
            y="medae",
            color="method",
            barmode="group",
            title="Masked validation: median absolute error by feature",
        )
        fig.update_layout(yaxis_title="MedAE in original units", xaxis_title="Feature", xaxis_tickangle=-25)
        snippets.append(_fig_to_html(fig))

        pivot = metrics.pivot_table(index="method", columns="feature", values="spearman", aggfunc="mean")
        if not pivot.empty:
            fig = px.imshow(
                pivot,
                zmin=-1,
                zmax=1,
                color_continuous_scale="RdBu_r",
                title="Masked validation: Spearman correlation, reconstructed vs available target",
                labels={"color": "Spearman"},
            )
            snippets.append(_fig_to_html(fig))

    predictions = method_result.validation_predictions.copy()
    if not predictions.empty:
        if len(predictions) > 2500:
            predictions = predictions.sample(2500, random_state=method_result.seed)
        fig = px.scatter(
            predictions,
            x="true_value",
            y="predicted_value",
            color="feature",
            facet_col="feature",
            facet_col_wrap=3,
            opacity=0.55,
            title=f"Masked validation reconstruction for {method_result.method}",
        )
        fig.update_layout(showlegend=False)
        snippets.append(_fig_to_html(fig))

    return "\n".join(snippets)


def _robust_plot_values(values: pd.Series, feature: str) -> tuple[pd.Series, str, str]:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if feature in POSITIVE_LOG_FEATURES:
        positive = numeric[numeric > 0]
        transformed = np.log10(positive)
        return transformed.replace([np.inf, -np.inf], np.nan).dropna(), f"log10({feature})", "finite positive values"
    return numeric, feature, "finite values"


def _clip_visual_range(values: pd.Series) -> tuple[pd.Series, str]:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite) < 3:
        return finite, "No percentile clipping applied."
    low, high = finite.quantile([0.005, 0.995])
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        return finite, "No percentile clipping applied."
    clipped = finite[(finite >= low) & (finite <= high)]
    note = f"Visual range clipped to p0.5-p99.5: [{low:.4g}, {high:.4g}]. Metrics use all values."
    return clipped, note


def _distribution_stage_frame(
    result: ImputationResult,
    method_result: MethodResult,
    feature: str,
) -> tuple[pd.DataFrame, str, str]:
    frames: list[pd.DataFrame] = []
    x_label = feature
    notes: list[str] = []
    for label, frame in [
        ("before physical/statistical completion", result.prepared_df),
        (f"after {method_result.method}", method_result.full_df),
    ]:
        if feature not in frame.columns:
            continue
        values, x_label, value_note = _robust_plot_values(frame[feature], feature)
        clipped, clip_note = _clip_visual_range(values)
        notes.append(f"{label}: {len(values):,} {value_note}; {clip_note}")
        if not clipped.empty:
            frames.append(pd.DataFrame({"stage": label, "value": clipped}))
    if not frames:
        return pd.DataFrame(), x_label, f"Insufficient finite positive values for this plot: {feature}"
    return pd.concat(frames, ignore_index=True), x_label, " ".join(notes)


def _distribution_source_frame(
    method_result: MethodResult,
    feature: str,
) -> tuple[pd.DataFrame, str, str]:
    source_col = f"{feature}_source"
    if feature not in method_result.full_df.columns or source_col not in method_result.full_df.columns:
        return pd.DataFrame(), feature, f"Insufficient finite positive values for this plot: {feature}"
    values, x_label, value_note = _robust_plot_values(method_result.full_df[feature], feature)
    source = method_result.full_df.loc[values.index, source_col].map(_source_category)
    frame = pd.DataFrame({"source": source, "value": values}).dropna()
    clipped, clip_note = _clip_visual_range(frame["value"])
    frame = frame.loc[clipped.index]
    if frame.empty:
        return pd.DataFrame(), x_label, f"Insufficient finite positive values for this plot: {feature}"
    counts = frame["source"].value_counts().to_dict()
    note = f"{len(values):,} {value_note}; plotted source counts: {counts}. {clip_note}"
    return frame, x_label, note


def _distribution_figures(result: ImputationResult, method_result: MethodResult) -> str:
    snippets: list[str] = []
    for feature in [feature for feature in REPORT_FEATURES if feature in result.features_included]:
        if feature not in result.prepared_df.columns or feature not in method_result.full_df.columns:
            continue
        plot_df, x_label, note = _distribution_stage_frame(result, method_result, feature)
        if plot_df.empty:
            snippets.append(f"<p><strong>{feature}</strong>: {note}</p>")
        else:
            nbins = max(12, min(55, int(math.sqrt(len(plot_df))) + 5))
            fig = px.histogram(plot_df, x="value", color="stage", barmode="overlay", nbins=nbins)
            fig.update_traces(opacity=0.55)
            fig.update_layout(
                title=f"Distribution before/after: {feature}",
                xaxis_title=x_label,
                yaxis_title="Rows",
                annotations=[
                    dict(text=note[:260], x=0, y=1.12, xref="paper", yref="paper", showarrow=False, align="left")
                ],
            )
            snippets.append(_fig_to_html(fig))

        source_df, source_x_label, source_note = _distribution_source_frame(method_result, feature)
        if source_df.empty:
            snippets.append(f"<p><strong>{feature} by source</strong>: {source_note}</p>")
        elif source_df["source"].nunique(dropna=True) > 1:
            nbins = max(12, min(55, int(math.sqrt(len(source_df))) + 5))
            fig = px.histogram(
                source_df,
                x="value",
                color="source",
                barmode="overlay",
                nbins=nbins,
                title=f"Final {feature} values by source",
            )
            fig.update_traces(opacity=0.55)
            fig.update_layout(
                xaxis_title=source_x_label,
                yaxis_title="Rows",
                annotations=[
                    dict(text=source_note[:260], x=0, y=1.12, xref="paper", yref="paper", showarrow=False, align="left")
                ],
            )
            snippets.append(_fig_to_html(fig))
    return "\n".join(snippets)


def _scatter_status(full_df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    columns = [x, y]
    for col in ["pl_name", "hostname", "discoverymethod", "disc_year"]:
        if col in full_df.columns:
            columns.append(col)
    for feature in [x, y]:
        source_col = f"{feature}_source"
        if source_col in full_df.columns:
            columns.append(source_col)
    plot_df = full_df.loc[:, list(dict.fromkeys(columns))].copy()
    plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna(subset=[x, y])
    statuses: list[str] = []
    for _, row in plot_df.iterrows():
        sources = [str(row.get(f"{feature}_source", "")) for feature in [x, y]]
        if any(source.startswith("imputed_") for source in sources):
            statuses.append("imputed in plotted pair")
        elif any(source.startswith("derived_") for source in sources):
            statuses.append("physically derived in plotted pair")
        else:
            statuses.append("observed in plotted pair")
    plot_df["value_status"] = statuses
    return plot_df


def _scatter_figures(method_result: MethodResult, config: ImputationConfig) -> str:
    snippets: list[str] = []
    for x, y, _, title in SCATTER_SPECS:
        if x not in method_result.full_df.columns or y not in method_result.full_df.columns:
            continue
        plot_df = _scatter_status(method_result.full_df, x, y)
        if plot_df.empty:
            continue
        if len(plot_df) > 3500:
            plot_df = plot_df.sample(3500, random_state=config.random_state)
        fig = px.scatter(
            plot_df,
            x=x,
            y=y,
            color="value_status",
            hover_name="pl_name" if "pl_name" in plot_df.columns else None,
            hover_data=[column for column in ["hostname", "discoverymethod", "disc_year"] if column in plot_df.columns],
            opacity=0.62,
            title=f"{title}, colored by imputation status ({method_result.method})",
        )
        if x in LOG10_FEATURES and (plot_df[x] > 0).all():
            fig.update_xaxes(type="log")
        if y in LOG10_FEATURES and (plot_df[y] > 0).all():
            fig.update_yaxes(type="log")
        snippets.append(_fig_to_html(fig))
    return "\n".join(snippets)


def build_report_html(
    result: ImputationResult,
    config: ImputationConfig,
    csv_path: Path,
    primary_key: str,
) -> str:
    method_result = result.methods[primary_key]
    sections: list[str] = []
    warning = (
        "Los valores imputados no son observaciones astron&oacute;micas. Sirven para an&aacute;lisis "
        "exploratorio, Mapper/TDA y sensibilidad estad&iacute;stica. Toda conclusi&oacute;n debe "
        "compararse contra casos completos y contra varios m&eacute;todos de imputaci&oacute;n."
    )
    css = """
    <style>
      body { font-family: Segoe UI, Arial, sans-serif; margin: 0; color: #17202a; background: #f7f8fa; }
      main { max-width: 1180px; margin: 0 auto; padding: 28px 24px 56px; }
      section { background: white; border: 1px solid #dde3ea; border-radius: 8px; padding: 18px; margin: 16px 0; }
      .warning { background: #fff4d6; border-color: #e2c66d; font-weight: 600; }
      .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
      .metric { background: #eef5f3; border: 1px solid #cbded8; border-radius: 6px; padding: 10px; }
      .metric strong { display: block; font-size: 21px; }
      .metric span { color: #566573; font-size: 12px; text-transform: uppercase; }
      .section-note { color: #566573; margin-top: -4px; }
      .data-table { border-collapse: collapse; width: 100%; font-size: 13px; }
      .table-wrap { overflow-x: auto; max-width: 100%; }
      .data-table th, .data-table td { border-bottom: 1px solid #e3e8ee; padding: 7px; text-align: left; max-width: 260px; overflow-wrap: anywhere; vertical-align: top; }
      .data-table th { background: #eef2f6; }
      code { background: #eef2f6; padding: 2px 5px; border-radius: 4px; }
    </style>
    """

    source_long = _source_counts_long(method_result)
    imputed_cells = int(source_long.loc[source_long["source"] == "imputed", "count"].sum()) if not source_long.empty else 0
    derived_cells = (
        int(source_long.loc[source_long["source"].eq("physically_derived"), "count"].sum())
        if not source_long.empty
        else 0
    )
    joint_row = method_result.feature_coverage[
        method_result.feature_coverage["feature_group"].eq("MAPPER_JOINT_FEATURES")
    ]
    joint_before = float(joint_row["before_complete_pct"].iloc[0]) if not joint_row.empty else np.nan
    joint_after = float(joint_row["after_complete_pct"].iloc[0]) if not joint_row.empty else np.nan

    sections.append(
        "<section><h2>Resumen del dataset</h2>"
        "<div class='grid'>"
        f"<div class='metric'><span>CSV</span><strong>{csv_path.name}</strong></div>"
        f"<div class='metric'><span>Filas</span><strong>{len(result.raw_df):,}</strong></div>"
        f"<div class='metric'><span>METODO VISUALIZADO</span><strong>{method_result.method}</strong></div>"
        f"<div class='metric'><span>Features Mapper</span><strong>{len(result.features_included)}</strong></div>"
        f"<div class='metric'><span>Celdas imputadas</span><strong>{imputed_cells:,}</strong></div>"
        f"<div class='metric'><span>Celdas derivadas</span><strong>{derived_cells:,}</strong></div>"
        f"<div class='metric'><span>Joint antes</span><strong>{joint_before:.1f}%</strong></div>"
        f"<div class='metric'><span>Joint despues</span><strong>{joint_after:.1f}%</strong></div>"
        "</div></section>"
    )
    sections.append(
        "<section class='warning'>"
        f"{warning}<br>"
        "Algunas variables, como densidad planetaria, pueden ser derivadas fisicamente desde otras mediciones; "
        "estas no son observaciones independientes. En este dataset, <code>pl_dens</code> se reconstruye "
        "principalmente mediante masa y radio. <code>iterative</code> se visualiza como metodo principal porque "
        "obtuvo el menor rank promedio de error en la validacion enmascarada entre los metodos evaluados. "
        "La estabilidad topologica debe compararse contra complete cases y metodos alternativos."
        "</section>"
    )

    sections.append(
        _report_section(
            "Variables used",
            f"<p><code>{', '.join(result.features_included)}</code></p>"
            "<h3>Excluded variables</h3>"
            f"{_html_table(result.excluded_features, 40)}",
            "These are the variables that entered the numerical imputation matrix after identifier/reference filters.",
        )
    )
    sections.append(
        _report_section(
            "Missingness Before And After",
            _missingness_fig(result, method_result) + _html_table(result.missingness_before, 30),
            "This separates raw missingness, values recovered by physical derivation, and values still needing imputation.",
        )
    )
    sections.append(
        _report_section(
            "Value Source Composition",
            _source_composition_fig(method_result) + _html_table(method_result.missingness_after, 30),
            "Observed values are kept; density and semimajor axis can be physically derived; remaining gaps are imputed.",
        )
    )
    sections.append(
        _report_section(
            "Mapper Coverage",
            _coverage_fig(method_result) + _html_table(method_result.feature_coverage, 20),
            "Coverage is shown for feature groups used in Mapper/TDA before and after the selected imputation method.",
        )
    )
    validation_html = _validation_figures(result, method_result)
    sections.append(
        _report_section(
            "Validation By Masking",
            validation_html + _html_table(method_result.validation_metrics, 40),
            "Derived/available values are hidden and reconstructed. Metrics identify whether validation targets were original observations or physically derived values.",
        )
    )
    if not result.method_comparison.empty:
        sections.append(
            _report_section(
                "Method Comparison",
                _method_comparison_fig(result.method_comparison) + _html_table(result.method_comparison, 10),
                "Lower validation ranks mean lower masked-validation error. This does not automatically settle topology.",
            )
        )

    sections.append(
        _report_section(
            "Distribution Checks",
            _distribution_figures(result, method_result),
            "Compare pre-imputation values with final values, then inspect final distributions by source.",
        )
    )
    sections.append(
        _report_section(
            "Science-Space Scatter Checks",
            _scatter_figures(method_result, config),
            "Points are colored by whether either plotted coordinate was observed, physically derived, or imputed.",
        )
    )

    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>Imputation report</title>"
        f"{css}</head><body><main><h1>PSCompPars imputation report</h1>"
        + "\n".join(sections)
        + "</main></body></html>"
    )


def _write_table_pair(df: pd.DataFrame, csv_path: Path, json_path: Path) -> dict[str, Path]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8")
    return {csv_path.stem: csv_path, f"{json_path.stem}_json": json_path}


def _all_metrics_table(result: ImputationResult) -> pd.DataFrame:
    frames = [method.validation_metrics for method in result.methods.values() if not method.validation_metrics.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _missingness_summary_table(result: ImputationResult, method_result: MethodResult) -> pd.DataFrame:
    return result.missingness_before.merge(
        method_result.missingness_after,
        on="feature",
        how="outer",
        suffixes=("_before", "_after"),
    )


def _value_source_composition_table(method_result: MethodResult) -> pd.DataFrame:
    cols = [
        "feature",
        "observed_count",
        "physically_derived_count",
        "imputed_count",
        "still_missing_count",
        "derived_density_count",
        "derived_kepler_count",
    ]
    available = [col for col in cols if col in method_result.missingness_after.columns]
    table = method_result.missingness_after[available].copy()
    table["method"] = method_result.method
    return table


def export_summary_tables(
    result: ImputationResult,
    method_result: MethodResult,
    outputs_dir: Path,
) -> dict[str, Path]:
    tables_dir = outputs_dir / "tables"
    paths: dict[str, Path] = {}
    table_specs = {
        "imputation_method_comparison": result.method_comparison,
        "imputation_validation_metrics": _all_metrics_table(result),
        "imputation_value_source_composition": _value_source_composition_table(method_result),
        "imputation_missingness_summary": _missingness_summary_table(result, method_result),
        "mapper_coverage_summary": method_result.feature_coverage,
    }
    for name, table in table_specs.items():
        paths.update(_write_table_pair(table, tables_dir / f"{name}.csv", tables_dir / f"{name}.json"))
    return paths


def _import_matplotlib_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_pdf(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"No se pudo exportar una figura PDF valida: {path}")


def _message_pdf(path: Path, title: str, message: str) -> None:
    plt = _import_matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.text(0.02, 0.7, message, transform=ax.transAxes, fontsize=11, wrap=True)
    _save_pdf(fig, path)
    plt.close(fig)


def _grouped_bar_pdf(df: pd.DataFrame, x: str, y: str, hue: str, title: str, ylabel: str, path: Path) -> None:
    if df.empty:
        _message_pdf(path, title, "No data available for this figure.")
        return
    plt = _import_matplotlib_pyplot()
    pivot = df.pivot_table(index=x, columns=hue, values=y, aggfunc="mean").fillna(0)
    fig, ax = plt.subplots(figsize=(13, 7))
    pivot.plot(kind="bar", ax=ax, width=0.82)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(4, max(1, len(pivot.columns))))
    _save_pdf(fig, path)
    plt.close(fig)


def _stacked_bar_pdf(df: pd.DataFrame, x: str, y: str, hue: str, title: str, ylabel: str, path: Path) -> None:
    if df.empty:
        _message_pdf(path, title, "No data available for this figure.")
        return
    plt = _import_matplotlib_pyplot()
    pivot = df.pivot_table(index=x, columns=hue, values=y, aggfunc="sum").fillna(0)
    fig, ax = plt.subplots(figsize=(13, 7))
    pivot.plot(kind="bar", stacked=True, ax=ax, width=0.82)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(4, max(1, len(pivot.columns))))
    _save_pdf(fig, path)
    plt.close(fig)


def _validation_spearman_pdf(metrics: pd.DataFrame, path: Path) -> None:
    title = "Masked validation Spearman correlation"
    if metrics.empty or "spearman" not in metrics.columns:
        _message_pdf(path, title, "No Spearman metrics available.")
        return
    pivot = metrics.pivot_table(index="method", columns="feature", values="spearman", aggfunc="mean")
    if pivot.empty:
        _message_pdf(path, title, "No Spearman metrics available.")
        return
    plt = _import_matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(12, 6))
    image = ax.imshow(pivot.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Spearman")
    _save_pdf(fig, path)
    plt.close(fig)


def _distribution_pdf(
    result: ImputationResult,
    method_result: MethodResult,
    feature: str,
    path: Path,
) -> None:
    title = f"Distribution by source: {feature}"
    frame, x_label, note = _distribution_source_frame(method_result, feature)
    if frame.empty:
        _message_pdf(path, title, note)
        return
    plt = _import_matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(12, 7))
    bins = max(12, min(55, int(math.sqrt(len(frame))) + 5))
    for source, group in frame.groupby("source"):
        ax.hist(group["value"], bins=bins, alpha=0.55, label=f"{source} (n={len(group):,})")
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Rows")
    ax.text(0.01, 0.98, note[:220], transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    _save_pdf(fig, path)
    plt.close(fig)


def _scatter_pdf(method_result: MethodResult, x: str, y: str, title: str, path: Path) -> None:
    plot_df = _scatter_status(method_result.full_df, x, y)
    if plot_df.empty:
        _message_pdf(path, title, "No finite values available for this scatter plot.")
        return
    plt = _import_matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(11, 8))
    x_values = plot_df[x]
    y_values = plot_df[y]
    x_label = x
    y_label = y
    if x in POSITIVE_LOG_FEATURES:
        mask = x_values > 0
        x_values = np.log10(x_values[mask])
        y_values = y_values[mask]
        plot_df = plot_df.loc[mask]
        x_label = f"log10({x})"
    if y in POSITIVE_LOG_FEATURES:
        mask = y_values > 0
        y_values = np.log10(y_values[mask])
        x_values = x_values[mask]
        plot_df = plot_df.loc[mask]
        y_label = f"log10({y})"
    if plot_df.empty:
        _message_pdf(path, title, "Insufficient finite positive values for this scatter plot.")
        plt.close(fig)
        return
    for status, idx in plot_df.groupby("value_status").groups.items():
        ax.scatter(x_values.loc[idx], y_values.loc[idx], s=18, alpha=0.58, label=f"{status} (n={len(idx):,})")
    ax.set_title(f"{title} ({method_result.method})", loc="left", fontsize=14, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    _save_pdf(fig, path)
    plt.close(fig)


def export_report_figures(
    result: ImputationResult,
    method_result: MethodResult,
    outputs_dir: Path,
) -> dict[str, Path]:
    figures_dir = outputs_dir / "figures_pdf"
    paths: dict[str, Path] = {}

    before = result.missingness_before[["feature", "missing_raw_pct", "missing_after_physical_derivation_pct"]]
    after = method_result.missingness_after[["feature", "missing_after_pct"]]
    missing_plot = before.merge(after, on="feature", how="outer").melt(
        id_vars=["feature"],
        var_name="stage",
        value_name="missing_pct",
    )
    missing_plot["stage"] = missing_plot["stage"].map(
        {
            "missing_raw_pct": "raw CSV",
            "missing_after_physical_derivation_pct": "after physical derivation",
            "missing_after_pct": f"after {method_result.method}",
        }
    )
    path = figures_dir / "01_missingness_before_after.pdf"
    _grouped_bar_pdf(missing_plot, "feature", "missing_pct", "stage", "Missingness before/after", "% missing", path)
    paths[path.stem] = path

    path = figures_dir / "02_value_source_composition.pdf"
    _stacked_bar_pdf(_source_counts_long(method_result), "feature", "count", "source", "Value source composition", "Rows", path)
    paths[path.stem] = path

    coverage_plot = method_result.feature_coverage.melt(
        id_vars=["feature_group"],
        value_vars=["before_complete_pct", "after_complete_pct"],
        var_name="stage",
        value_name="complete_pct",
    )
    coverage_plot["stage"] = coverage_plot["stage"].map(
        {"before_complete_pct": "before", "after_complete_pct": "after"}
    )
    path = figures_dir / "03_mapper_coverage.pdf"
    _grouped_bar_pdf(coverage_plot, "feature_group", "complete_pct", "stage", "Mapper feature coverage", "% complete rows", path)
    paths[path.stem] = path

    metrics = _all_metrics_table(result)
    path = figures_dir / "04_masked_validation_mae_by_feature.pdf"
    _grouped_bar_pdf(metrics, "feature", "mae", "method", "Masked validation MAE by feature", "MAE", path)
    paths[path.stem] = path

    path = figures_dir / "05_masked_validation_spearman_heatmap.pdf"
    _validation_spearman_pdf(metrics, path)
    paths[path.stem] = path

    comparison_plot = result.method_comparison.melt(
        id_vars=["method"],
        value_vars=["mean_mae_rank", "mean_rmse_rank"],
        var_name="rank_type",
        value_name="mean_rank",
    )
    path = figures_dir / "06_method_comparison.pdf"
    _grouped_bar_pdf(comparison_plot, "method", "mean_rank", "rank_type", "Method comparison ranks", "Mean rank", path)
    paths[path.stem] = path

    for offset, feature in enumerate(REPORT_FEATURES, start=7):
        path = figures_dir / f"{offset:02d}_distribution_{feature}.pdf"
        _distribution_pdf(result, method_result, feature, path)
        paths[path.stem] = path

    scatter_names = [
        "14_scatter_mass_radius.pdf",
        "15_scatter_density_radius.pdf",
        "16_scatter_orbper_orbsmax.pdf",
        "17_scatter_insol_eqt.pdf",
    ]
    for (x, y, _, title), filename in zip(SCATTER_SPECS, scatter_names):
        path = figures_dir / filename
        _scatter_pdf(method_result, x, y, title, path)
        paths[path.stem] = path

    return paths


def export_report_outputs(
    result: ImputationResult,
    method_result: MethodResult,
    outputs_dir: Path,
    export_figures: bool = True,
) -> dict[str, Path]:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    paths = export_summary_tables(result, method_result, outputs_dir)
    if export_figures:
        paths.update(export_report_figures(result, method_result, outputs_dir))
    return paths


def write_imputation_outputs(
    result: ImputationResult,
    config: ImputationConfig,
    csv_path: Path,
    reports_dir: Path,
    outputs_dir: Path | None = None,
    export_figures: bool = True,
) -> dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    result.missingness_before.to_csv(reports_dir / "missingness_profile_before.csv", index=False)
    paths["missingness_profile_before"] = reports_dir / "missingness_profile_before.csv"
    result.log_transform_audit.to_csv(reports_dir / "log_transform_audit.csv", index=False)
    paths["log_transform_audit"] = reports_dir / "log_transform_audit.csv"
    result.excluded_features.to_csv(reports_dir / "excluded_features.csv", index=False)
    paths["excluded_features"] = reports_dir / "excluded_features.csv"
    pd.DataFrame(
        [
            {
                **{f"density_{key}": value for key, value in asdict(result.physical_audit.density).items()},
                **{f"kepler_{key}": value for key, value in asdict(result.physical_audit.kepler).items()},
            }
        ]
    ).to_csv(reports_dir / "physical_derivations.csv", index=False)
    paths["physical_derivations"] = reports_dir / "physical_derivations.csv"

    for key, method_result in result.methods.items():
        suffix = method_result.method if key == method_result.method else key
        full_path = reports_dir / f"PSCompPars_imputed_{suffix}.csv"
        mapper_path = reports_dir / f"mapper_features_imputed_{suffix}.csv"
        metrics_path = reports_dir / f"validation_metrics_{suffix}.csv"
        predictions_path = reports_dir / f"validation_predictions_long_{suffix}.csv"
        after_path = reports_dir / f"missingness_profile_after_{suffix}.csv"
        coverage_path = reports_dir / f"feature_coverage_before_after_{suffix}.csv"
        values_path = reports_dir / f"imputed_values_long_{suffix}.csv"
        summary_path = reports_dir / f"imputation_summary_{suffix}.json"

        method_result.full_df.to_csv(full_path, index=False)
        method_result.mapper_tables["joint_imputed"].to_csv(mapper_path, index=False)
        method_result.validation_metrics.to_csv(metrics_path, index=False)
        method_result.validation_predictions.to_csv(predictions_path, index=False)
        method_result.missingness_after.to_csv(after_path, index=False)
        method_result.feature_coverage.to_csv(coverage_path, index=False)
        method_result.imputed_values_long.to_csv(values_path, index=False)
        write_json(summary_path, method_result.summary)

        paths[f"PSCompPars_imputed_{suffix}"] = full_path
        paths[f"mapper_features_imputed_{suffix}"] = mapper_path
        paths[f"validation_metrics_{suffix}"] = metrics_path
        paths[f"validation_predictions_long_{suffix}"] = predictions_path
        paths[f"missingness_profile_after_{suffix}"] = after_path
        paths[f"feature_coverage_before_after_{suffix}"] = coverage_path
        paths[f"imputed_values_long_{suffix}"] = values_path
        paths[f"imputation_summary_{suffix}"] = summary_path

    primary_key = select_visualized_key(result, config.visualized_method)
    result.methods[primary_key].mapper_tables["joint_complete"].to_csv(
        reports_dir / "mapper_features_complete_case.csv",
        index=False,
    )
    paths["mapper_features_complete_case"] = reports_dir / "mapper_features_complete_case.csv"

    if "median" in result.methods:
        paths.setdefault("PSCompPars_imputed_median", reports_dir / "PSCompPars_imputed_median.csv")
    if "knn" in result.methods:
        paths.setdefault("PSCompPars_imputed_knn", reports_dir / "PSCompPars_imputed_knn.csv")
    if "iterative" in result.methods:
        paths.setdefault("PSCompPars_imputed_iterative", reports_dir / "PSCompPars_imputed_iterative.csv")

    comparison_path = reports_dir / "method_comparison.csv"
    result.method_comparison.to_csv(comparison_path, index=False)
    paths["method_comparison"] = comparison_path

    config_path = reports_dir / "imputation_config.json"
    write_json(
        config_path,
        {
            "csv_path": str(csv_path),
            "method": config.method,
            "visualized_method": config.visualized_method,
            "n_neighbors": config.n_neighbors,
            "weights": config.weights,
            "max_missing_pct": config.max_missing_pct,
            "validation_mask_frac": config.validation_mask_frac,
            "random_state": config.random_state,
            "n_multiple_imputations": config.n_multiple_imputations,
            "include_stellar_context": config.include_stellar_context,
            "include_orbital_eccentricity": config.include_orbital_eccentricity,
            "features_requested": result.features_requested,
            "features_included": result.features_included,
            "excluded_features": result.excluded_features.to_dict(orient="records"),
        },
    )
    paths["imputation_config"] = config_path

    report_path = reports_dir / "imputation_report.html"
    report_path.write_text(
        build_report_html(result, config, csv_path, primary_key),
        encoding="utf-8",
    )
    paths["imputation_report"] = report_path
    export_dir = outputs_dir or reports_dir / "outputs"
    paths.update(export_report_outputs(result, result.methods[primary_key], export_dir, export_figures=export_figures))
    return paths
