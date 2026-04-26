from __future__ import annotations

import json
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


GRAPH_DISTANCE_COLUMNS = [
    "n_nodes",
    "n_edges",
    "beta_0",
    "beta_1",
    "graph_density",
    "average_degree",
    "average_clustering",
    "diameter_largest_component",
    "average_shortest_path_largest_component",
]


def _python_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _json_list(values: list[Any]) -> str:
    return json.dumps([_python_scalar(value) for value in values], ensure_ascii=False)


def _label_entropy(labels: pd.Series) -> float:
    counts = labels.value_counts(normalize=True, dropna=False)
    if counts.empty:
        return 0.0
    return float(-(counts * np.log2(counts)).sum())


def _dominant_label(labels: pd.Series) -> tuple[str | None, float | None]:
    nonmissing = labels.dropna()
    if nonmissing.empty:
        return None, None
    counts = nonmissing.value_counts(normalize=True)
    label = str(counts.index[0])
    pct = float(counts.iloc[0] * 100.0)
    return label, pct


def assign_physical_family(row: pd.Series) -> str:
    radius = pd.to_numeric(pd.Series([row.get("pl_rade")]), errors="coerce").iloc[0]
    mass = pd.to_numeric(pd.Series([row.get("pl_bmasse")]), errors="coerce").iloc[0]

    if pd.notna(mass) and mass > 1000:
        return "super_jupiter"
    if pd.notna(radius) and radius >= 12.0:
        return "super_jupiter"
    if pd.isna(radius):
        return "unknown"
    if radius < 1.25:
        return "terrestrial"
    if radius < 2.0:
        return "super_earth"
    if radius < 4.0:
        return "sub_neptune"
    if radius < 12.0:
        return "gas_giant"
    return "unknown"


def assign_orbital_class(row: pd.Series) -> str:
    eqt = pd.to_numeric(pd.Series([row.get("pl_eqt")]), errors="coerce").iloc[0]
    orbper = pd.to_numeric(pd.Series([row.get("pl_orbper")]), errors="coerce").iloc[0]
    orbsmax = pd.to_numeric(pd.Series([row.get("pl_orbsmax")]), errors="coerce").iloc[0]

    if pd.notna(eqt) and eqt >= 1000:
        return "hot"
    if pd.notna(orbper) and orbper < 10:
        return "hot"
    if pd.notna(eqt) and 300 <= eqt < 1000:
        return "warm"
    if pd.notna(eqt) and 180 <= eqt < 300:
        return "temperate"
    if pd.notna(eqt) and eqt < 180:
        return "cold_or_distant"
    if pd.notna(orbsmax) and orbsmax > 1.5:
        return "cold_or_distant"
    return "unknown"


def mapper_graph_to_networkx(graph: dict) -> nx.Graph:
    nx_graph = nx.Graph()
    nodes = graph.get("nodes", {})
    for node_id, members in nodes.items():
        nx_graph.add_node(node_id, size=len(members), sample_indices=list(members))
    for source, targets in graph.get("links", {}).items():
        if source not in nx_graph:
            nx_graph.add_node(source, size=len(nodes.get(source, [])), sample_indices=list(nodes.get(source, [])))
        for target in targets:
            if target not in nx_graph:
                nx_graph.add_node(target, size=len(nodes.get(target, [])), sample_indices=list(nodes.get(target, [])))
            if source != target:
                nx_graph.add_edge(source, target)
    return nx_graph


def compute_graph_metrics(nx_graph: nx.Graph, graph: dict) -> dict:
    node_sizes = [len(members) for members in graph.get("nodes", {}).values()]
    n_nodes = int(nx_graph.number_of_nodes())
    n_edges = int(nx_graph.number_of_edges())
    beta_0 = int(nx.number_connected_components(nx_graph)) if n_nodes else 0
    beta_1 = int(n_edges - n_nodes + beta_0)

    largest_component_n_nodes = 0
    largest_component_n_edges = 0
    diameter = None
    average_shortest_path = None
    if n_nodes:
        components = list(nx.connected_components(nx_graph))
        if components:
            largest_nodes = max(components, key=len)
            subgraph = nx_graph.subgraph(largest_nodes).copy()
            largest_component_n_nodes = int(subgraph.number_of_nodes())
            largest_component_n_edges = int(subgraph.number_of_edges())
            if largest_component_n_nodes == 1:
                diameter = 0
                average_shortest_path = 0.0
            elif largest_component_n_nodes > 1:
                try:
                    diameter = int(nx.diameter(subgraph))
                except Exception:
                    diameter = None
                try:
                    average_shortest_path = float(nx.average_shortest_path_length(subgraph))
                except Exception:
                    average_shortest_path = None

    metrics = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "beta_0": beta_0,
        "beta_1": beta_1,
        "graph_density": float(nx.density(nx_graph)) if n_nodes else 0.0,
        "average_degree": float(np.mean([degree for _, degree in nx_graph.degree()])) if n_nodes else 0.0,
        "n_connected_components": beta_0,
        "largest_component_n_nodes": largest_component_n_nodes,
        "largest_component_n_edges": largest_component_n_edges,
        "diameter_largest_component": diameter,
        "average_shortest_path_largest_component": average_shortest_path,
        "average_clustering": float(nx.average_clustering(nx_graph)) if n_nodes else 0.0,
        "mean_node_size": float(np.mean(node_sizes)) if node_sizes else 0.0,
        "median_node_size": float(np.median(node_sizes)) if node_sizes else 0.0,
        "max_node_size": int(np.max(node_sizes)) if node_sizes else 0,
    }
    return metrics


def _sample_ids_for_members(graph: dict, members: list[int]) -> list[Any]:
    lookup = graph.get("sample_id_lookup")
    if lookup:
        return [_python_scalar(lookup[index]) for index in members]
    return [_python_scalar(index) for index in members]


def build_node_table(
    graph: dict,
    work_df: pd.DataFrame,
    used_features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    nodes = graph.get("nodes", {})

    for node_id, members in nodes.items():
        node_df = work_df.iloc[members].copy()
        row: dict[str, Any] = {
            "node_id": node_id,
            "n_points": int(len(members)),
            "sample_indices": _json_list(_sample_ids_for_members(graph, list(members))),
        }
        if "pl_name" in node_df.columns:
            row["pl_name_list"] = _json_list(sorted(node_df["pl_name"].dropna().astype(str).unique().tolist()))
        if "hostname" in node_df.columns:
            row["hostname_list"] = _json_list(sorted(node_df["hostname"].dropna().astype(str).unique().tolist()))

        for feature in used_features:
            if feature not in node_df.columns:
                continue
            values = pd.to_numeric(node_df[feature], errors="coerce")
            row[f"{feature}_mean"] = float(values.mean()) if values.notna().any() else None
            row[f"{feature}_median"] = float(values.median()) if values.notna().any() else None

        if "discoverymethod" in node_df.columns:
            dominant, pct = _dominant_label(node_df["discoverymethod"].astype("string"))
            row["discoverymethod_dominant"] = dominant
            row["discoverymethod_dominant_pct"] = pct
        if "disc_year" in node_df.columns:
            disc_year = pd.to_numeric(node_df["disc_year"], errors="coerce")
            row["disc_year_median"] = float(disc_year.median()) if disc_year.notna().any() else None
        if "sy_dist" in node_df.columns:
            sy_dist = pd.to_numeric(node_df["sy_dist"], errors="coerce")
            row["sy_dist_median"] = float(sy_dist.median()) if sy_dist.notna().any() else None

        missing_flag_columns = [f"{feature}_was_missing" for feature in used_features if f"{feature}_was_missing" in node_df.columns]
        if missing_flag_columns:
            flag_values = node_df.loc[:, missing_flag_columns].astype(float).to_numpy()
            row["imputed_missing_fraction_mean"] = float(np.nanmean(flag_values))
        else:
            row["imputed_missing_fraction_mean"] = None

        physical_labels = node_df.apply(assign_physical_family, axis=1)
        orbital_labels = node_df.apply(assign_orbital_class, axis=1)
        physical_dominant, physical_pct = _dominant_label(physical_labels.astype("string"))
        orbital_dominant, orbital_pct = _dominant_label(orbital_labels.astype("string"))
        row["physical_family_dominant"] = physical_dominant
        row["physical_family_dominant_pct"] = physical_pct
        row["physical_family_entropy"] = _label_entropy(physical_labels.astype("string"))
        row["orbital_class_dominant"] = orbital_dominant
        row["orbital_class_dominant_pct"] = orbital_pct
        row["orbital_class_entropy"] = _label_entropy(orbital_labels.astype("string"))

        rows.append(row)

    return pd.DataFrame(rows)


def build_edge_table(graph: dict) -> pd.DataFrame:
    nodes = graph.get("nodes", {})
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []

    for source, targets in graph.get("links", {}).items():
        source_members = set(nodes.get(source, []))
        for target in targets:
            edge = tuple(sorted((source, target)))
            if source == target or edge in seen:
                continue
            seen.add(edge)
            target_members = set(nodes.get(target, []))
            shared_positions = sorted(source_members & target_members)
            shared_ids = _sample_ids_for_members(graph, shared_positions)
            rows.append(
                {
                    "source": edge[0],
                    "target": edge[1],
                    "shared_points_count": int(len(shared_positions)),
                    "shared_sample_indices": _json_list(shared_ids),
                }
            )

    return pd.DataFrame(rows)


def _standardize_metric_columns(metrics_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    standardized = pd.DataFrame(index=metrics_df.index)
    for column in columns:
        values = pd.to_numeric(metrics_df[column], errors="coerce")
        mean = values.mean(skipna=True)
        std = values.std(skipna=True, ddof=0)
        if pd.isna(std) or std == 0:
            standardized[column] = values.where(values.isna(), 0.0)
        else:
            standardized[column] = (values - mean) / std
    return standardized


def _distance_record(
    frame: pd.DataFrame,
    standardized: pd.DataFrame,
    left_index: Any,
    right_index: Any,
    label: str,
) -> dict[str, Any]:
    common_columns = [
        column
        for column in standardized.columns
        if pd.notna(standardized.loc[left_index, column]) and pd.notna(standardized.loc[right_index, column])
    ]
    if not common_columns:
        distance = None
    else:
        delta = standardized.loc[left_index, common_columns] - standardized.loc[right_index, common_columns]
        distance = float(np.linalg.norm(delta.to_numpy(dtype=float)))

    left = frame.loc[left_index]
    right = frame.loc[right_index]
    return {
        "comparison": label,
        "graph_a": left["config_id"],
        "graph_b": right["config_id"],
        "space_a": left.get("space"),
        "lens_a": left.get("lens"),
        "space_b": right.get("space"),
        "lens_b": right.get("lens"),
        "n_cubes": left.get("n_cubes"),
        "overlap": left.get("overlap"),
        "distance_l2": distance,
        "common_metric_count": int(len(common_columns)),
        "metrics_used": ", ".join(common_columns),
    }


def compare_mapper_graphs(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df is None or metrics_df.empty or len(metrics_df) < 2:
        return pd.DataFrame()

    frame = metrics_df.copy()
    if "config_id" not in frame.columns:
        frame["config_id"] = [f"graph_{index:02d}" for index in range(len(frame))]

    metric_columns = [column for column in GRAPH_DISTANCE_COLUMNS if column in frame.columns]
    if not metric_columns:
        return pd.DataFrame()

    standardized = _standardize_metric_columns(frame, metric_columns)
    pair_specs = [
        ("phys", "pca2", "orb", "pca2", "phys_pca2_vs_orb_pca2"),
        ("phys", "pca2", "joint", "pca2", "phys_pca2_vs_joint_pca2"),
        ("orb", "pca2", "joint", "pca2", "orb_pca2_vs_joint_pca2"),
        ("phys", "density", "orb", "density", "phys_density_vs_orb_density"),
        ("phys", "density", "joint", "density", "phys_density_vs_joint_density"),
        ("orb", "density", "joint", "density", "orb_density_vs_joint_density"),
        ("phys", "pca2", "phys", "density", "phys_pca2_vs_phys_density"),
        ("orb", "pca2", "orb", "density", "orb_pca2_vs_orb_density"),
        ("joint", "pca2", "joint", "density", "joint_pca2_vs_joint_density"),
    ]

    group_columns = [column for column in ["n_cubes", "overlap"] if column in frame.columns]
    groups = (
        frame[group_columns].drop_duplicates().to_dict(orient="records") if group_columns else [dict()]
    )

    rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[Any, Any, str]] = set()

    for group in groups:
        if group_columns:
            mask = pd.Series(True, index=frame.index)
            for column, value in group.items():
                mask &= frame[column] == value
            subset = frame.loc[mask]
        else:
            subset = frame

        by_key = {
            (row["space"], row["lens"]): index
            for index, row in subset[["space", "lens"]].iterrows()
        }
        for left_space, left_lens, right_space, right_lens, label in pair_specs:
            left_index = by_key.get((left_space, left_lens))
            right_index = by_key.get((right_space, right_lens))
            if left_index is None or right_index is None:
                continue
            pair_key = tuple(sorted((left_index, right_index))) + (label,)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            rows.append(_distance_record(frame, standardized, left_index, right_index, label))

    if rows:
        return pd.DataFrame(rows)

    fallback_rows = [
        _distance_record(frame, standardized, left_index, right_index, "pairwise_fallback")
        for left_index, right_index in combinations(frame.index.tolist(), 2)
    ]
    return pd.DataFrame(fallback_rows)
