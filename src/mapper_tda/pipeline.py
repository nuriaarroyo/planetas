from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from feature_config import (
    LOG10_FEATURES,
    MAPPER_JOINT_FEATURES,
    MAPPER_ORB_FEATURES,
    MAPPER_ORB_OPTIONAL_FEATURES,
    MAPPER_PHYS_FEATURES,
)
from imputation.steps.physical_derivation import apply_physical_derivations
from .cluster import estimate_dbscan_eps, make_clusterer
from .lenses import make_lens_density, make_lens_domain, make_lens_pca2
from .metrics import build_edge_table, build_node_table, compare_mapper_graphs, compute_graph_metrics, mapper_graph_to_networkx
from .preprocessing import preprocess_mapper_features


N_CUBES_GRID = [6, 8, 10, 12, 15]
OVERLAP_GRID = [0.20, 0.30, 0.35, 0.40, 0.50]


@dataclass
class MapperConfig:
    space: str = "joint"
    lens: str = "pca2"
    n_cubes: int = 10
    overlap: float = 0.35
    clusterer: str = "dbscan"
    min_samples: int = 4
    eps_percentile: float = 90
    k_density: int = 15
    complete_case_only: bool = True
    include_eccentricity: bool = False
    random_state: int = 42


def config_id(config: MapperConfig) -> str:
    overlap_slug = str(config.overlap).replace(".", "p")
    return f"{config.space}_{config.lens}_cubes{config.n_cubes}_overlap{overlap_slug}"


def select_features_for_space(config: MapperConfig) -> list[str]:
    if config.space == "phys":
        return list(MAPPER_PHYS_FEATURES)
    if config.space == "orb":
        features = list(MAPPER_ORB_FEATURES)
        if config.include_eccentricity:
            features.extend(MAPPER_ORB_OPTIONAL_FEATURES)
        return list(dict.fromkeys(features))
    if config.space == "joint":
        features = list(MAPPER_JOINT_FEATURES)
        if config.include_eccentricity:
            features.extend(MAPPER_ORB_OPTIONAL_FEATURES)
        return list(dict.fromkeys(features))
    raise ValueError(f"Espacio no soportado: {config.space}")


def _canonicalize_source_dataframe(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    prepared = df.copy()
    for feature in set(features) | {"pl_rade", "pl_bmasse", "pl_dens", "pl_orbper", "pl_orbsmax", "st_mass"}:
        original_column = f"original_{feature}"
        if original_column in prepared.columns:
            prepared[feature] = pd.to_numeric(prepared[original_column], errors="coerce")

    physical_audit = None
    if any(feature in set(features) for feature in {"pl_dens", "pl_orbsmax"}):
        prepared, physical_audit = apply_physical_derivations(prepared)
        for feature in ("pl_dens", "pl_orbsmax"):
            original_column = f"original_{feature}"
            if original_column in prepared.columns:
                original_values = pd.to_numeric(prepared[original_column], errors="coerce")
                prepared[original_column] = original_values.where(original_values.notna(), prepared[feature])

    return prepared, {
        "physical_derivations": {
            "density": asdict(physical_audit.density) if physical_audit else None,
            "kepler": asdict(physical_audit.kepler) if physical_audit else None,
        }
    }


def _lens_names(config: MapperConfig, lens_metadata: dict[str, Any]) -> list[str]:
    if config.lens == "pca2":
        return ["PC1", "PC2"]
    if config.lens == "density":
        return ["PC1", "log_d_k"]
    if config.lens == "domain":
        columns = lens_metadata.get("columns", [])
        return [str(column) for column in columns]
    return ["lens_1", "lens_2"]


def _default_color_column(space: str, work_df: pd.DataFrame, used_features: list[str]) -> str | None:
    preferred = {
        "phys": "pl_rade",
        "orb": "pl_orbper",
        "joint": "pl_rade",
    }.get(space)
    if preferred and preferred in work_df.columns:
        return preferred
    for column in ["pl_bmasse", "pl_dens", "pl_insol", "pl_eqt"]:
        if column in work_df.columns:
            return column
    return used_features[0] if used_features else None


def _visualize_mapper_graph(
    mapper: Any,
    graph: dict[str, Any],
    config: MapperConfig,
    work_df: pd.DataFrame,
    Z: np.ndarray,
    used_features: list[str],
    lens: np.ndarray,
    lens_metadata: dict[str, Any],
) -> tuple[str, str | None]:
    if not graph.get("nodes"):
        html = (
            "<!doctype html><html><head><meta charset='utf-8'><title>Mapper vacio</title></head>"
            "<body><h1>Mapper graph vacío</h1><p>La configuración no produjo nodos.</p></body></html>"
        )
        return html, None

    color_column = _default_color_column(config.space, work_df, used_features)
    if color_column is None:
        html = (
            "<!doctype html><html><head><meta charset='utf-8'><title>Mapper sin color</title></head>"
            "<body><h1>Mapper graph</h1><p>No se encontró una columna numérica adecuada para colorear.</p></body></html>"
        )
        return html, None

    color_values = pd.to_numeric(work_df[color_column], errors="coerce")
    if color_values.isna().all():
        color_values = pd.Series(np.zeros(len(work_df)), index=work_df.index)
    else:
        color_values = color_values.fillna(float(color_values.median()))

    tooltip_series = None
    if "pl_name" in work_df.columns:
        tooltip_series = work_df["pl_name"].astype("string")
    elif "hostname" in work_df.columns:
        tooltip_series = work_df["hostname"].astype("string")

    try:
        html = mapper.visualize(
            graph,
            color_values=color_values.to_numpy(dtype=float).reshape(-1, 1),
            color_function_name=[color_column],
            custom_tooltips=tooltip_series.fillna("").to_numpy() if tooltip_series is not None else None,
            save_file=False,
            X=Z,
            X_names=used_features,
            lens=lens,
            lens_names=_lens_names(config, lens_metadata),
            title=f"Mapper {config.space} / {config.lens}",
            include_searchbar=True,
            include_min_intersection_selector=True,
        )
    except Exception as exc:
        html = (
            "<!doctype html><html><head><meta charset='utf-8'><title>Mapper error</title></head>"
            f"<body><h1>No se pudo renderizar el HTML interactivo</h1><p>{exc}</p></body></html>"
        )
    return html, color_column


def build_mapper_graph(
    df: pd.DataFrame,
    config: MapperConfig,
) -> dict:
    """
    Build one Mapper graph for one space and one lens.

    Returns dictionary with:
    - graph
    - nx_graph
    - lens
    - work_df
    - Z
    - used_features
    - lens_metadata
    - mapper_metadata
    - node_table
    - edge_table
    - graph_metrics
    """

    try:
        import kmapper as km
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar 'kmapper'. Instala la dependencia con `pip install kmapper>=2.1`."
        ) from exc

    features = select_features_for_space(config)
    source_df, source_metadata = _canonicalize_source_dataframe(df, features)
    work_df, Z, scaler, used_features = preprocess_mapper_features(
        source_df,
        features=features,
        log10_features=list(LOG10_FEATURES),
        complete_case_only=config.complete_case_only,
    )

    if config.lens == "pca2":
        lens, lens_metadata = make_lens_pca2(Z, random_state=config.random_state)
    elif config.lens == "density":
        lens, lens_metadata = make_lens_density(Z, k_density=config.k_density, random_state=config.random_state)
    elif config.lens == "domain":
        lens, lens_metadata = make_lens_domain(work_df, space=config.space)
    else:
        raise ValueError(f"Lens no soportado: {config.lens}")

    mapper = km.KeplerMapper()
    estimated_eps = estimate_dbscan_eps(Z, min_samples=config.min_samples, percentile=config.eps_percentile)
    clusterer = make_clusterer(
        clusterer=config.clusterer,
        Z=Z,
        min_samples=config.min_samples,
        eps_percentile=config.eps_percentile,
    )
    graph = mapper.map(
        lens,
        X=Z,
        clusterer=clusterer,
        cover=km.Cover(n_cubes=config.n_cubes, perc_overlap=config.overlap),
    )
    graph["sample_id_lookup"] = [_python_scalar(index) for index in work_df.index.tolist()]

    nx_graph = mapper_graph_to_networkx(graph)
    graph_metrics = compute_graph_metrics(nx_graph, graph)
    node_table = build_node_table(graph, work_df, used_features)
    edge_table = build_edge_table(graph)
    html, color_column = _visualize_mapper_graph(
        mapper=mapper,
        graph=graph,
        config=config,
        work_df=work_df,
        Z=Z,
        used_features=used_features,
        lens=lens,
        lens_metadata=lens_metadata,
    )

    mapper_metadata = {
        "config_id": config_id(config),
        "space": config.space,
        "lens": config.lens,
        "n_rows_input": int(len(df)),
        "n_rows_used": int(len(work_df)),
        "used_features": used_features,
        "cover": {"n_cubes": int(config.n_cubes), "overlap": float(config.overlap)},
        "clusterer": {
            "name": config.clusterer,
            "min_samples": int(config.min_samples),
            "eps_percentile": float(config.eps_percentile),
            "estimated_eps": float(estimated_eps),
        },
        "lens_metadata": lens_metadata,
        "preprocessing": work_df.attrs.get("preprocessing", {}),
        "physical_derivations": source_metadata.get("physical_derivations"),
        "kepler_mapper_meta": graph.get("meta_data", {}),
    }

    return {
        "config": config,
        "config_id": config_id(config),
        "graph": graph,
        "nx_graph": nx_graph,
        "lens": lens,
        "work_df": work_df,
        "Z": Z,
        "scaler": scaler,
        "used_features": used_features,
        "lens_metadata": lens_metadata,
        "mapper_metadata": mapper_metadata,
        "node_table": node_table,
        "edge_table": edge_table,
        "graph_metrics": graph_metrics,
        "html": html,
        "color_column": color_column,
    }


def _python_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def expand_configs_from_cli(args) -> list[MapperConfig]:
    spaces = ["phys", "orb", "joint"] if args.space == "all" else [args.space]
    lenses = ["pca2", "density", "domain"] if args.lens == "all" else [args.lens]
    n_cubes_values = N_CUBES_GRID if args.grid else [args.n_cubes]
    overlap_values = OVERLAP_GRID if args.grid else [args.overlap]

    configs: list[MapperConfig] = []
    for space in spaces:
        for lens in lenses:
            for n_cubes in n_cubes_values:
                for overlap in overlap_values:
                    configs.append(
                        MapperConfig(
                            space=space,
                            lens=lens,
                            n_cubes=n_cubes,
                            overlap=overlap,
                            clusterer=args.clusterer,
                            min_samples=args.min_samples,
                            eps_percentile=args.eps_percentile,
                            k_density=args.k_density,
                            complete_case_only=args.complete_case_only,
                            include_eccentricity=args.include_eccentricity,
                            random_state=args.random_state,
                        )
                    )
    return configs


def _metrics_row(result: dict) -> dict[str, Any]:
    config: MapperConfig = result["config"]
    row = {
        "config_id": result["config_id"],
        "space": config.space,
        "lens": config.lens,
        "n_cubes": int(config.n_cubes),
        "overlap": float(config.overlap),
        "clusterer": config.clusterer,
        "min_samples": int(config.min_samples),
        "eps_percentile": float(config.eps_percentile),
        "k_density": int(config.k_density),
        "rows_used": int(len(result["work_df"])),
        "n_features": int(len(result["used_features"])),
        "color_column": result.get("color_column"),
    }
    row.update(result["graph_metrics"])
    return row


def build_stability_grid(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty or not {"space", "lens"}.issubset(metrics_df.columns):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (space, lens), group in metrics_df.groupby(["space", "lens"], dropna=False):
        rows.append(
            {
                "space": space,
                "lens": lens,
                "n_runs": int(len(group)),
                "n_nodes_min": int(group["n_nodes"].min()) if "n_nodes" in group else None,
                "n_nodes_max": int(group["n_nodes"].max()) if "n_nodes" in group else None,
                "n_edges_min": int(group["n_edges"].min()) if "n_edges" in group else None,
                "n_edges_max": int(group["n_edges"].max()) if "n_edges" in group else None,
                "beta_1_min": int(group["beta_1"].min()) if "beta_1" in group else None,
                "beta_1_max": int(group["beta_1"].max()) if "beta_1" in group else None,
                "average_degree_min": float(group["average_degree"].min()) if "average_degree" in group else None,
                "average_degree_max": float(group["average_degree"].max()) if "average_degree" in group else None,
                "average_clustering_min": float(group["average_clustering"].min())
                if "average_clustering" in group
                else None,
                "average_clustering_max": float(group["average_clustering"].max())
                if "average_clustering" in group
                else None,
            }
        )
    return pd.DataFrame(rows)


def run_mapper_batch(
    df: pd.DataFrame,
    configs: list[MapperConfig],
    csv_path: Path,
    grid_mode: bool = False,
) -> dict[str, Any]:
    results = [build_mapper_graph(df, config) for config in configs]
    metrics_df = pd.DataFrame([_metrics_row(result) for result in results]).sort_values(
        ["space", "lens", "n_cubes", "overlap"],
        ignore_index=True,
    )
    distances_df = compare_mapper_graphs(metrics_df)
    stability_grid_df = build_stability_grid(metrics_df) if grid_mode else pd.DataFrame()

    return {
        "csv_path": csv_path,
        "results": results,
        "metrics_df": metrics_df,
        "distances_df": distances_df,
        "stability_grid_df": stability_grid_df,
        "config_summary": {
            "dataset": {
                "csv_path": str(csv_path),
                "csv_name": csv_path.name,
                "rows_input": int(len(df)),
                "grid_mode": bool(grid_mode),
            },
            "configs": [
                {
                    **asdict(config),
                    "config_id": config_id(config),
                }
                for config in configs
            ],
        },
    }
