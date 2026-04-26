from __future__ import annotations

from dataclasses import asdict, dataclass, field
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


@dataclass(frozen=True)
class ImputationConfig:
    method: str = "knn"
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

    for feature in excluded:
        if feature not in out.columns:
            continue
        source_col = f"{feature}_source"
        out[source_col] = "excluded_too_missing"
        raw_missing = raw_df[feature].isna() if feature in raw_df.columns else pd.Series(True, index=out.index)
        out[f"{feature}_was_missing"] = raw_missing.astype(bool).to_numpy()

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
                "imputed_count": int(source.astype("string").str.startswith("imputed_", na=False).sum()),
                "excluded_too_missing_count": int((source == "excluded_too_missing").sum()),
                "still_missing_count": missing,
            }
        )
    return pd.DataFrame(rows)


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


def validation_metric_row(method: str, feature: str, truth: pd.Series, prediction: pd.Series) -> dict[str, Any]:
    errors = prediction - truth
    abs_errors = errors.abs()
    return {
        "method": method,
        "feature": feature,
        "n_validated": int(len(truth)),
        "mae": float(abs_errors.mean()) if len(abs_errors) else np.nan,
        "rmse": float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else np.nan,
        "medae": float(abs_errors.median()) if len(abs_errors) else np.nan,
        "mape": _mape_or_nan(truth, prediction),
        "spearman": float(truth.corr(prediction, method="spearman"))
        if len(truth) >= 2 and truth.nunique() > 1 and prediction.nunique() > 1
        else np.nan,
        "pearson": float(truth.corr(prediction, method="pearson"))
        if len(truth) >= 2 and truth.nunique() > 1 and prediction.nunique() > 1
        else np.nan,
    }


def run_masked_validation(
    numeric: pd.DataFrame,
    method: str,
    config: ImputationConfig,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    transformed, _ = apply_log10_transform(numeric, default_log_features(list(numeric.columns)))
    masked = numeric.copy()
    rng = np.random.default_rng(seed)
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
        masked.loc[chosen, feature] = np.nan

    imputed, _, _, _, _ = impute_numeric_matrix(masked, method, config, seed)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for feature, idx in mask_by_feature.items():
        if len(idx) == 0:
            continue
        truth = numeric.loc[idx, feature]
        prediction = imputed.loc[idx, feature]
        metric_rows.append(validation_metric_row(method, feature, truth, prediction))
        for row_index in idx:
            true_value = numeric.loc[row_index, feature]
            predicted_value = imputed.loc[row_index, feature]
            prediction_rows.append(
                {
                    "method": method,
                    "feature": feature,
                    "row_index": row_index,
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
            after = missingness_profile_after(full_df, features, excluded_features, method)
            coverage = coverage_table(prepared_df, full_df, features, config, method)
            validation_metrics, validation_predictions = run_masked_validation(numeric, method, config, seed)
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
    return df.head(max_rows).to_html(index=False, border=0, classes="data-table")


def _fig_to_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def build_report_html(
    result: ImputationResult,
    config: ImputationConfig,
    csv_path: Path,
    primary_key: str,
) -> str:
    method_result = result.methods[primary_key]
    sections: list[str] = []
    warning = (
        "Los valores imputados no son observaciones astronómicas. Sirven para análisis exploratorio, "
        "Mapper/TDA y sensibilidad estadística. Toda conclusión debe compararse contra casos completos "
        "y contra varios métodos de imputación."
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
      .data-table { border-collapse: collapse; width: 100%; font-size: 13px; }
      .data-table th, .data-table td { border-bottom: 1px solid #e3e8ee; padding: 7px; text-align: left; }
      code { background: #eef2f6; padding: 2px 5px; border-radius: 4px; }
    </style>
    """
    sections.append(
        "<section><h2>Resumen del dataset</h2>"
        "<div class='grid'>"
        f"<div class='metric'><span>CSV</span><strong>{csv_path.name}</strong></div>"
        f"<div class='metric'><span>Filas</span><strong>{len(result.raw_df):,}</strong></div>"
        f"<div class='metric'><span>Metodo</span><strong>{method_result.method}</strong></div>"
        f"<div class='metric'><span>Features</span><strong>{len(result.features_included)}</strong></div>"
        "</div></section>"
    )
    sections.append(f"<section class='warning'>{warning}</section>")
    sections.append(
        "<section><h2>Variables usadas</h2>"
        f"<p><code>{', '.join(result.features_included)}</code></p>"
        "<h3>Variables excluidas por exceso de missing u otras reglas</h3>"
        f"{_html_table(result.excluded_features, 40)}</section>"
    )
    sections.append(
        "<section><h2>Cobertura antes/despues</h2>"
        f"{_html_table(method_result.feature_coverage, 20)}</section>"
    )
    sections.append(
        "<section><h2>Metricas de validacion</h2>"
        f"{_html_table(method_result.validation_metrics, 40)}</section>"
    )
    if not result.method_comparison.empty:
        sections.append(
            "<section><h2>Comparacion de metodos</h2>"
            f"{_html_table(result.method_comparison, 10)}</section>"
        )

    hist_features = ["pl_rade", "pl_bmasse", "pl_dens", "pl_orbper", "pl_orbsmax", "pl_insol", "pl_eqt"]
    for feature in hist_features:
        if feature in result.prepared_df.columns and feature in method_result.full_df.columns:
            plot_df = pd.DataFrame(
                {
                    "before": pd.to_numeric(result.prepared_df[feature], errors="coerce"),
                    "after": pd.to_numeric(method_result.full_df[feature], errors="coerce"),
                }
            ).melt(var_name="stage", value_name=feature)
            plot_df = plot_df.dropna()
            if not plot_df.empty:
                fig = px.histogram(plot_df, x=feature, color="stage", barmode="overlay", nbins=50)
                fig.update_traces(opacity=0.55)
                fig.update_layout(title=f"Histograma antes/despues: {feature}")
                sections.append(f"<section>{_fig_to_html(fig)}</section>")

    scatter_pairs = [
        ("pl_bmasse", "pl_rade"),
        ("pl_dens", "pl_rade"),
        ("pl_orbper", "pl_rade"),
        ("pl_insol", "pl_eqt"),
    ]
    for x, y in scatter_pairs:
        if x in method_result.full_df.columns and y in method_result.full_df.columns:
            plot_df = method_result.full_df[[x, y]].copy()
            plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
            plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
            plot_df = plot_df.dropna()
            if not plot_df.empty:
                if len(plot_df) > 3000:
                    plot_df = plot_df.sample(3000, random_state=config.random_state)
                fig = px.scatter(plot_df, x=x, y=y, opacity=0.55, title=f"{x} vs {y}")
                sections.append(f"<section>{_fig_to_html(fig)}</section>")

    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>Imputation report</title>"
        f"{css}</head><body><main><h1>PSCompPars imputation report</h1>"
        + "\n".join(sections)
        + "</main></body></html>"
    )


def write_imputation_outputs(
    result: ImputationResult,
    config: ImputationConfig,
    csv_path: Path,
    reports_dir: Path,
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

    primary_key = "knn" if "knn" in result.methods else next(iter(result.methods))
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
    return paths
