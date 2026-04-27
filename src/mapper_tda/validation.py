from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import align_mapper_and_physical_inputs, load_csv, resolve_mapper_features_path, resolve_physical_csv_path
from .pipeline import MapperConfig, build_mapper_graph


MAIN_VALIDATION_CONFIGS = [
    "phys_min_pca2_cubes10_overlap0p35",
    "orbital_pca2_cubes10_overlap0p35",
    "joint_no_density_pca2_cubes10_overlap0p35",
    "joint_pca2_cubes10_overlap0p35",
    "thermal_pca2_cubes10_overlap0p35",
]


def _select_validation_results(batch_result: dict[str, Any], include_thermal: bool = True) -> list[dict[str, Any]]:
    allowed = set(MAIN_VALIDATION_CONFIGS if include_thermal else MAIN_VALIDATION_CONFIGS[:-1])
    return [result for result in batch_result["results"] if result["config_id"] in allowed]


def run_bootstrap_validation(
    batch_result: dict[str, Any],
    n_bootstrap: int = 30,
    bootstrap_frac: float = 0.8,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    for result in _select_validation_results(batch_result, include_thermal=True):
        mapper_df = result["mapper_df"].reset_index(drop=True)
        physical_df = result["physical_df"].reset_index(drop=True)
        config: MapperConfig = result["config"]
        sample_size = max(3, int(len(mapper_df) * bootstrap_frac))
        for iteration in range(n_bootstrap):
            indices = rng.choice(len(mapper_df), size=sample_size, replace=True)
            boot_result = build_mapper_graph(mapper_df.iloc[indices].reset_index(drop=True), physical_df.iloc[indices].reset_index(drop=True), config)
            rows.append(
                {
                    "config_id": result["config_id"],
                    "bootstrap_iter": iteration,
                    "n_nodes": boot_result["graph_metrics"]["n_nodes"],
                    "n_edges": boot_result["graph_metrics"]["n_edges"],
                    "beta_0": boot_result["graph_metrics"]["beta_0"],
                    "beta_1": boot_result["graph_metrics"]["beta_1"],
                    "mean_node_imputation_fraction": float(pd.to_numeric(boot_result["node_table"]["mean_imputation_fraction"], errors="coerce").mean()) if not boot_result["node_table"].empty else 0.0,
                    "largest_component_fraction": boot_result["graph_metrics"]["largest_component_fraction"],
                }
            )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics, pd.DataFrame()
    summary_rows: list[dict[str, Any]] = []
    for (config_id, metric), group in metrics.melt(id_vars=["config_id", "bootstrap_iter"], var_name="metric", value_name="value").groupby(["config_id", "metric"]):
        values = pd.to_numeric(group["value"], errors="coerce").dropna()
        summary_rows.append(
            {
                "config_id": config_id,
                "metric": metric,
                "mean": float(values.mean()) if not values.empty else np.nan,
                "std": float(values.std(ddof=0)) if not values.empty else np.nan,
                "cv": float(values.std(ddof=0) / values.mean()) if not values.empty and values.mean() not in [0, np.nan] else np.nan,
                "q05": float(values.quantile(0.05)) if not values.empty else np.nan,
                "q50": float(values.quantile(0.50)) if not values.empty else np.nan,
                "q95": float(values.quantile(0.95)) if not values.empty else np.nan,
            }
        )
    return metrics, pd.DataFrame(summary_rows)


def run_null_models(
    batch_result: dict[str, Any],
    n_null: int = 30,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    for result in _select_validation_results(batch_result, include_thermal=False):
        mapper_df = result["mapper_df"].reset_index(drop=True)
        physical_df = result["physical_df"].reset_index(drop=True)
        config: MapperConfig = result["config"]
        for iteration in range(n_null):
            shuffled = mapper_df.copy()
            for column in [column for column in result["used_features"] if column in shuffled.columns]:
                shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            null_result = build_mapper_graph(shuffled, physical_df, config)
            rows.append(
                {
                    "config_id": result["config_id"],
                    "null_iter": iteration,
                    "null_model": "column_shuffle",
                    "n_nodes": null_result["graph_metrics"]["n_nodes"],
                    "n_edges": null_result["graph_metrics"]["n_edges"],
                    "beta_0": null_result["graph_metrics"]["beta_0"],
                    "beta_1": null_result["graph_metrics"]["beta_1"],
                }
            )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics, pd.DataFrame()
    summary_rows: list[dict[str, Any]] = []
    real_lookup = batch_result["metrics_df"].set_index("config_id")
    for (config_id, metric), group in metrics.melt(id_vars=["config_id", "null_iter", "null_model"], var_name="metric", value_name="value").groupby(["config_id", "metric"]):
        values = pd.to_numeric(group["value"], errors="coerce").dropna()
        real_value = float(real_lookup.loc[config_id, metric]) if config_id in real_lookup.index and metric in real_lookup.columns else np.nan
        null_mean = float(values.mean()) if not values.empty else np.nan
        null_std = float(values.std(ddof=0)) if not values.empty else np.nan
        empirical_p_high = float((values >= real_value).mean()) if not values.empty and pd.notna(real_value) else np.nan
        z_score = float((real_value - null_mean) / null_std) if pd.notna(real_value) and pd.notna(null_std) and null_std > 0 else np.nan
        summary_rows.append(
            {
                "config_id": config_id,
                "metric": metric,
                "real_value": real_value,
                "null_mean": null_mean,
                "null_std": null_std,
                "empirical_p_high": empirical_p_high,
                "z_score_vs_null": z_score,
            }
        )
    return metrics, pd.DataFrame(summary_rows)


def run_imputation_method_comparison(
    batch_result: dict[str, Any],
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del random_state
    config_lookup = {result["config_id"]: result["config"] for result in batch_result["results"]}
    target_config_ids = [
        "phys_min_pca2_cubes10_overlap0p35",
        "orbital_pca2_cubes10_overlap0p35",
        "joint_no_density_pca2_cubes10_overlap0p35",
        "joint_pca2_cubes10_overlap0p35",
    ]
    rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []
    for method in ["iterative", "knn", "median", "complete_case"]:
        try:
            mapper_path = resolve_mapper_features_path(input_method=method)
            physical_path = resolve_physical_csv_path(input_method=method)
            mapper_df, physical_df, _ = align_mapper_and_physical_inputs(load_csv(mapper_path), load_csv(physical_path))
            availability_rows.append({"input_method": method, "available": True, "mapper_features_path": str(mapper_path), "physical_csv_path": str(physical_path)})
        except Exception:
            availability_rows.append({"input_method": method, "available": False, "mapper_features_path": "", "physical_csv_path": ""})
            continue

        for config_id in target_config_ids:
            if config_id not in config_lookup:
                continue
            config = config_lookup[config_id]
            compare_config = MapperConfig(**config.__dict__)
            compare_config.input_method = method
            result = build_mapper_graph(mapper_df, physical_df, compare_config)
            mean_imputation = float(pd.to_numeric(result["node_table"]["mean_imputation_fraction"], errors="coerce").mean()) if not result["node_table"].empty else 0.0
            rows.append(
                {
                    "input_method": method,
                    "config_id": config_id,
                    "space": compare_config.space,
                    "lens": compare_config.lens,
                    "n_nodes": result["graph_metrics"]["n_nodes"],
                    "n_edges": result["graph_metrics"]["n_edges"],
                    "beta_0": result["graph_metrics"]["beta_0"],
                    "beta_1": result["graph_metrics"]["beta_1"],
                    "mean_node_imputation_fraction": mean_imputation,
                    "largest_component_fraction": result["graph_metrics"]["largest_component_fraction"],
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(availability_rows)
