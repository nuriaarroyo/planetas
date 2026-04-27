from __future__ import annotations

import json
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


MAIN_GRAPH_CONFIGS = {
    "phys_min_pca2_cubes10_overlap0p35": ("Control masa-radio sin densidad derivada.", "primary", "low"),
    "phys_density_pca2_cubes10_overlap0p35": ("Mide el efecto de agregar densidad derivada.", "control", "moderate"),
    "orbital_pca2_cubes10_overlap0p35": ("Espacio orbital con baja imputacion y alta prioridad interpretativa.", "primary", "low"),
    "joint_no_density_pca2_cubes10_overlap0p35": ("Espacio conjunto sin densidad derivada redundante.", "primary", "moderate"),
    "joint_pca2_cubes10_overlap0p35": ("Espacio conjunto completo para comparar el efecto de pl_dens.", "control", "moderate"),
    "thermal_pca2_cubes10_overlap0p35": ("Alta complejidad, pero interpretacion debilitada por imputacion.", "cautionary", "high"),
}


def _parse_member_indices(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    if pd.isna(value):
        return []
    try:
        parsed = json.loads(value)
        return [int(v) for v in parsed]
    except Exception:
        return []


def build_main_graph_selection(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_id = metrics_df.set_index("config_id")
    for config_id, (reason, priority, caution) in MAIN_GRAPH_CONFIGS.items():
        if config_id not in by_id.index:
            continue
        row = by_id.loc[config_id]
        rows.append(
            {
                "config_id": config_id,
                "reason_for_selection": reason,
                "interpretation_priority": priority,
                "caution_level": caution,
                "n_nodes": row.get("n_nodes"),
                "n_edges": row.get("n_edges"),
                "beta_1": row.get("beta_1"),
                "mean_node_imputation_fraction": row.get("mean_node_imputation_fraction"),
                "frac_nodes_high_imputation": row.get("frac_nodes_high_imputation"),
                "mean_node_physically_derived_fraction": row.get("mean_node_physically_derived_fraction"),
            }
        )
    return pd.DataFrame(rows)


def _component_stats_for_graph(nx_graph: nx.Graph) -> dict[int, dict[str, Any]]:
    stats: dict[int, dict[str, Any]] = {}
    for component_id, component_nodes in enumerate(nx.connected_components(nx_graph)):
        subgraph = nx_graph.subgraph(component_nodes).copy()
        v = subgraph.number_of_nodes()
        e = subgraph.number_of_edges()
        beta_1_component = int(e - v + 1) if v else 0
        if beta_1_component < 0:
            beta_1_component = 0
        stats[component_id] = {
            "nodes": set(component_nodes),
            "n_nodes": int(v),
            "n_edges_internal": int(e),
            "beta_1_component": beta_1_component,
        }
    return stats


def _join_examples(series: pd.Series, limit: int = 5) -> str:
    values = []
    for item in series.dropna().head(limit).tolist():
        values.append(str(item))
    return ", ".join(values)


def _node_interpretation_text(row: pd.Series) -> tuple[str, str]:
    texts: list[str] = []
    cautions: list[str] = []
    population = str(row.get("candidate_population_top") or "")
    orbit_top = str(row.get("orbit_class_top") or "")
    thermal_top = str(row.get("thermal_class_top") or "")
    mean_imputation = float(pd.to_numeric(pd.Series([row.get("mean_imputation_fraction")]), errors="coerce").fillna(0).iloc[0])
    derived_fraction = float(pd.to_numeric(pd.Series([row.get("physically_derived_fraction")]), errors="coerce").fillna(0).iloc[0])

    if population == "hot_jupiter_candidate" and mean_imputation < 0.15:
        texts.append("Node enriched in short-period large-radius planets, compatible with a hot-Jupiter-like region.")
    if orbit_top == "long_period" and mean_imputation < 0.15:
        texts.append("Node concentrated in long-period orbital configurations with low imputation dependence.")
    if thermal_top in {"very_hot", "hot"} and mean_imputation >= 0.30:
        texts.append("Thermally extreme node, but interpretation is weakened by high imputation dependence.")
    if not texts:
        texts.append("Node summarizes a heuristic physical region induced by the completed Mapper feature space.")

    if derived_fraction > 0.30:
        cautions.append(
            "Node is influenced by physically derived quantities; interpret density-related structure as algebraically informed rather than independent."
        )
    if mean_imputation >= 0.30 or float(row.get("frac_any_imputed", 0.0)) >= 0.50:
        cautions.append("High imputation dependence reduces confidence in fine-grained scientific interpretation.")
    if not cautions:
        cautions.append("Low-to-moderate caution: interpretation is comparatively stable within this run.")
    return " ".join(texts), " ".join(cautions)


def build_highlighted_nodes(result: dict[str, Any]) -> pd.DataFrame:
    node_table = result["node_table"].copy()
    if node_table.empty:
        return pd.DataFrame()
    nx_graph = result["nx_graph"]
    component_stats = _component_stats_for_graph(nx_graph)

    size_q = node_table["n_members"].quantile(0.9)
    degree_q = node_table["degree"].quantile(0.9)
    thresholds = {
        "mean_pl_rade_high": node_table["mean_pl_rade"].quantile(0.95) if "mean_pl_rade" in node_table else np.nan,
        "mean_pl_rade_low": node_table["mean_pl_rade"].quantile(0.05) if "mean_pl_rade" in node_table else np.nan,
        "mean_pl_bmasse_high": node_table["mean_pl_bmasse"].quantile(0.95) if "mean_pl_bmasse" in node_table else np.nan,
        "mean_pl_orbper_high": node_table["mean_pl_orbper"].quantile(0.95) if "mean_pl_orbper" in node_table else np.nan,
        "mean_pl_eqt_high": node_table["mean_pl_eqt"].quantile(0.95) if "mean_pl_eqt" in node_table else np.nan,
        "mean_pl_insol_high": node_table["mean_pl_insol"].quantile(0.95) if "mean_pl_insol" in node_table else np.nan,
        "mean_pl_dens_high": node_table["mean_pl_dens"].quantile(0.95) if "mean_pl_dens" in node_table else np.nan,
    }
    rows: list[dict[str, Any]] = []
    for _, row in node_table.iterrows():
        reasons: list[str] = []
        component_info = component_stats.get(int(row["component_id"]), {"beta_1_component": 0})
        if row["n_members"] >= size_q:
            reasons.append("large_node")
        if row["degree"] >= degree_q:
            reasons.append("central_node")
        if component_info.get("beta_1_component", 0) > 0 and row["degree"] >= 2:
            reasons.append("cycle_component_node")
        if "mean_pl_rade" in row and pd.notna(row["mean_pl_rade"]):
            if row["mean_pl_rade"] >= thresholds["mean_pl_rade_high"]:
                reasons.append("extreme_large_radius")
            if row["mean_pl_rade"] <= thresholds["mean_pl_rade_low"]:
                reasons.append("extreme_small_radius")
        for column, reason in [
            ("mean_pl_bmasse", "extreme_mass"),
            ("mean_pl_orbper", "extreme_orbital_period"),
            ("mean_pl_eqt", "extreme_temperature"),
            ("mean_pl_insol", "extreme_insolation"),
            ("mean_pl_dens", "extreme_density"),
        ]:
            if column in row and pd.notna(row[column]) and row[column] >= thresholds.get(f"{column}_high", np.nan):
                reasons.append(reason)
        if float(row.get("mean_imputation_fraction", 1.0)) < 0.15 and float(row.get("frac_any_imputed", 1.0)) < 0.30:
            reasons.append("reliable_node")
        if float(row.get("mean_imputation_fraction", 0.0)) >= 0.30 or float(row.get("frac_any_imputed", 0.0)) >= 0.50:
            reasons.append("risky_node")
        if float(row.get("physically_derived_fraction", 0.0)) >= 0.30:
            reasons.append("derived_dominated_node")
        if not reasons:
            continue

        interpretation_text, caution_text = _node_interpretation_text(row)
        rows.append(
            {
                "config_id": row["config_id"],
                "node_id": row["node_id"],
                "highlight_reason": ";".join(sorted(set(reasons))),
                "n_members": row["n_members"],
                "degree": row["degree"],
                "component_id": row["component_id"],
                "mean_imputation_fraction": row.get("mean_imputation_fraction"),
                "physically_derived_fraction": row.get("physically_derived_fraction"),
                "candidate_population_top": row.get("candidate_population_top"),
                "radius_class_top": row.get("radius_class_top"),
                "orbit_class_top": row.get("orbit_class_top"),
                "thermal_class_top": row.get("thermal_class_top"),
                "mean_pl_rade": row.get("mean_pl_rade"),
                "mean_pl_bmasse": row.get("mean_pl_bmasse"),
                "mean_pl_dens": row.get("mean_pl_dens"),
                "mean_pl_orbper": row.get("mean_pl_orbper"),
                "mean_pl_orbsmax": row.get("mean_pl_orbsmax"),
                "mean_pl_insol": row.get("mean_pl_insol"),
                "mean_pl_eqt": row.get("mean_pl_eqt"),
                "example_pl_names": row.get("example_pl_names"),
                "interpretation_text": interpretation_text,
                "caution_text": caution_text,
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.sort_values(["config_id", "n_members", "degree"], ascending=[True, False, False]).reset_index(drop=True)


def build_component_summary(result: dict[str, Any]) -> pd.DataFrame:
    node_table = result["node_table"].copy()
    physical_df = result["physical_df"].copy()
    nx_graph = result["nx_graph"]
    if node_table.empty:
        return pd.DataFrame()
    component_stats = _component_stats_for_graph(nx_graph)
    rows: list[dict[str, Any]] = []
    for component_id, info in component_stats.items():
        component_nodes = node_table[node_table["component_id"] == component_id].copy()
        if component_nodes.empty:
            continue
        member_indices: list[int] = []
        for value in component_nodes["member_indices"]:
            member_indices.extend(_parse_member_indices(value))
        member_indices = sorted(set(member_indices))
        member_frame = physical_df.iloc[member_indices].copy() if member_indices else physical_df.iloc[[]].copy()

        dominant_radius = component_nodes["radius_class_top"].astype("string").value_counts().index[0] if "radius_class_top" in component_nodes and not component_nodes["radius_class_top"].dropna().empty else None
        dominant_orbit = component_nodes["orbit_class_top"].astype("string").value_counts().index[0] if "orbit_class_top" in component_nodes and not component_nodes["orbit_class_top"].dropna().empty else None
        dominant_thermal = component_nodes["thermal_class_top"].astype("string").value_counts().index[0] if "thermal_class_top" in component_nodes and not component_nodes["thermal_class_top"].dropna().empty else None
        dominant_population = component_nodes["candidate_population_top"].astype("string").value_counts().index[0] if "candidate_population_top" in component_nodes and not component_nodes["candidate_population_top"].dropna().empty else None
        mean_imputation = float(pd.to_numeric(component_nodes["mean_imputation_fraction"], errors="coerce").mean())
        derived_fraction = float(pd.to_numeric(component_nodes["physically_derived_fraction"], errors="coerce").mean())
        if mean_imputation >= 0.30:
            caution = "high"
        elif mean_imputation < 0.15:
            caution = "low"
        else:
            caution = "moderate"
        interpretation_text = (
            f"Connected component with {int(info['n_nodes'])} nodes and beta_1_component={int(info['beta_1_component'])}. "
            f"Dominant heuristic population: {dominant_population or 'not available'}."
        )
        rows.append(
            {
                "config_id": result["config_id"],
                "component_id": component_id,
                "n_nodes": info["n_nodes"],
                "n_members_unique": int(len(member_indices)),
                "n_edges_internal": info["n_edges_internal"],
                "beta_1_component": info["beta_1_component"],
                "mean_node_size": float(pd.to_numeric(component_nodes["n_members"], errors="coerce").mean()),
                "mean_imputation_fraction": mean_imputation,
                "physically_derived_fraction": derived_fraction,
                "dominant_radius_class": dominant_radius,
                "dominant_orbit_class": dominant_orbit,
                "dominant_thermal_class": dominant_thermal,
                "dominant_candidate_population": dominant_population,
                "median_pl_rade": float(pd.to_numeric(member_frame.get("pl_rade"), errors="coerce").median()) if "pl_rade" in member_frame else np.nan,
                "median_pl_bmasse": float(pd.to_numeric(member_frame.get("pl_bmasse"), errors="coerce").median()) if "pl_bmasse" in member_frame else np.nan,
                "median_pl_orbper": float(pd.to_numeric(member_frame.get("pl_orbper"), errors="coerce").median()) if "pl_orbper" in member_frame else np.nan,
                "median_pl_eqt": float(pd.to_numeric(member_frame.get("pl_eqt"), errors="coerce").median()) if "pl_eqt" in member_frame else np.nan,
                "example_nodes": ", ".join(component_nodes["node_id"].astype(str).head(5).tolist()),
                "example_pl_names": _join_examples(member_frame.get("pl_name", pd.Series(dtype="string"))),
                "interpretation_text": interpretation_text,
                "caution_level": caution,
            }
        )
    return pd.DataFrame(rows).sort_values(["config_id", "component_id"]).reset_index(drop=True)
