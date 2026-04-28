from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:  # pragma: no cover - optional report aid
    go = None
    make_subplots = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAPPER_ROOT = PROJECT_ROOT / "outputs" / "mapper"
GRAPH_DIR = MAPPER_ROOT / "graphs"
NODE_DIR = MAPPER_ROOT / "nodes"
EDGE_DIR = MAPPER_ROOT / "edges"
TABLE_DIR = MAPPER_ROOT / "tables"
LATEX_FIGURE_DIR = PROJECT_ROOT / "latex" / "03_mapper" / "figures"
INTERACTIVE_DIR = MAPPER_ROOT / "interactive"
MANIFEST_PATH = TABLE_DIR / "graph_render_manifest.md"


SELECTED_CONFIGS = [
    "phys_min_pca2_cubes10_overlap0p35",
    "phys_density_pca2_cubes10_overlap0p35",
    "orbital_pca2_cubes10_overlap0p35",
    "joint_no_density_pca2_cubes10_overlap0p35",
    "joint_pca2_cubes10_overlap0p35",
    "thermal_pca2_cubes10_overlap0p35",
]

SHORT_TITLES = {
    "phys_min_pca2_cubes10_overlap0p35": "Physical\nmass–radius",
    "phys_density_pca2_cubes10_overlap0p35": "Physical + density",
    "orbital_pca2_cubes10_overlap0p35": "Orbital",
    "joint_no_density_pca2_cubes10_overlap0p35": "Joint\nno density",
    "joint_pca2_cubes10_overlap0p35": "Joint + density",
    "thermal_pca2_cubes10_overlap0p35": "Thermal",
}

EVIDENCE_COLORS = {
    "physical": "#2E7D32",
    "observational": "#C62828",
    "mixed": "#F39C12",
    "weak": "#9E9E9E",
    "unclassified": "#D8D8D8",
}

METHOD_COLORS = {
    "Transit": "#2B6CB0",
    "Radial Velocity": "#6B46C1",
    "Imaging": "#C53030",
    "Microlensing": "#8B5E34",
    "Transit Timing Variations": "#00897B",
    "Eclipse Timing Variations": "#7B1FA2",
    "Pulsar Timing": "#455A64",
    "Astrometry": "#D81B60",
    "Orbital Brightness Modulation": "#F9A825",
    "Disk Kinematics": "#546E7A",
    "unknown": "#BDBDBD",
}

RADIUS_COLORS = {
    "rocky_size": "#1F77B4",
    "sub_neptune_size": "#2CA02C",
    "neptune_or_sub_jovian_size": "#FF7F0E",
    "jovian_size": "#D62728",
    "unknown": "#BDBDBD",
}

ORBIT_COLORS = {
    "short_period": "#E45756",
    "intermediate_period": "#4C78A8",
    "long_period": "#54A24B",
    "unknown": "#BDBDBD",
}

THERMAL_COLORS = {
    "very_hot": "#B2182B",
    "hot": "#EF8A62",
    "warm": "#67A9CF",
    "cool": "#2166AC",
    "unknown": "#BDBDBD",
}

CANDIDATE_POPULATION_COLORS = {
    "hot_jupiter_candidate": "#D62728",
    "warm_or_cool_giant_candidate": "#9467BD",
    "super_earth_candidate": "#1F77B4",
    "sub_neptune_candidate": "#2CA02C",
    "rocky_candidate": "#17BECF",
    "long_period_giant_candidate": "#8C564B",
    "unknown_mixed": "#9E9E9E",
    "unknown": "#BDBDBD",
}

ASTRO_CATEGORY_SPECS = {
    "candidate_population_plot": (
        "candidate_population_plot",
        CANDIDATE_POPULATION_COLORS,
        "unknown_mixed",
        "Candidate population",
    ),
    "radius_class_plot": ("radius_class_plot", RADIUS_COLORS, "unknown", "Radius class"),
    "orbit_class_plot": ("orbit_class_plot", ORBIT_COLORS, "unknown", "Orbit class"),
    "thermal_class_plot": ("thermal_class_plot", THERMAL_COLORS, "unknown", "Thermal class"),
}


@dataclass
class GraphRenderData:
    config_id: str
    title: str
    graph: nx.Graph
    nodes: pd.DataFrame
    positions: dict[str, tuple[float, float]]
    layout_used: str
    graph_path: Path
    node_path: Path
    edge_path: Path


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required input: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def graph_path(config_id: str) -> Path:
    return GRAPH_DIR / f"graph_{config_id}.json"


def node_path(config_id: str) -> Path:
    return NODE_DIR / f"nodes_{config_id}.csv"


def edge_path(config_id: str) -> Path:
    return EDGE_DIR / f"edges_{config_id}.csv"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def selected_configs() -> list[str]:
    selection_path = TABLE_DIR / "main_graph_selection.csv"
    if not selection_path.exists():
        return SELECTED_CONFIGS
    selection = read_csv(selection_path)
    present = [config for config in SELECTED_CONFIGS if config in set(selection["config_id"].astype(str))]
    return present or SELECTED_CONFIGS


def load_auxiliary_tables() -> dict[str, pd.DataFrame]:
    return {
        "synthesis": read_csv(TABLE_DIR / "final_region_synthesis.csv", required=False),
        "node_bias": read_csv(TABLE_DIR / "node_discovery_bias.csv", required=False),
        "enrichment": read_csv(TABLE_DIR / "discoverymethod_enrichment_summary.csv", required=False),
        "main_selection": read_csv(TABLE_DIR / "main_graph_selection.csv", required=False),
    }


def _merge_node_column(
    nodes: pd.DataFrame,
    source: pd.DataFrame,
    source_column: str,
    target_column: str | None = None,
) -> None:
    if source.empty or source_column not in source.columns:
        return
    target = target_column or source_column
    mapping = source.drop_duplicates("node_id").set_index("node_id")[source_column]
    values = nodes["node_id"].map(mapping)
    if target in nodes.columns:
        nodes[target] = nodes[target].combine_first(values)
    else:
        nodes[target] = values


def enrich_nodes(config_id: str, nodes: pd.DataFrame, aux: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = nodes.copy()
    out["node_id"] = out["node_id"].astype(str)

    synthesis = aux["synthesis"]
    if not synthesis.empty:
        node_regions = synthesis[
            (synthesis["config_id"].astype(str) == config_id)
            & (synthesis["region_type"].astype(str) == "node")
        ].copy()
        if "node_id" not in node_regions.columns or node_regions["node_id"].isna().all():
            node_regions["node_id"] = node_regions.get("region_id", pd.Series(dtype="string")).astype(str)
        for column in [
            "final_label",
            "confidence",
            "physical_evidence_score",
            "observational_bias_score",
            "imputation_risk_score",
            "dominant_discoverymethod",
            "dominant_discoverymethod_fraction",
            "discoverymethod_enrichment_z",
            "discoverymethod_enrichment_p",
            "radius_class_dominant",
            "orbit_class_dominant",
            "thermal_class_dominant",
        ]:
            _merge_node_column(out, node_regions, column)

    node_bias = aux["node_bias"]
    if not node_bias.empty:
        bias_rows = node_bias[node_bias["config_id"].astype(str) == config_id].copy()
        for column in [
            "dominant_discoverymethod",
            "dominant_discoverymethod_fraction",
            "discoverymethod_entropy",
            "discoverymethod_js_divergence_vs_global",
            "dominant_disc_facility",
            "dominant_disc_facility_fraction",
            "disc_year_median",
            "mean_imputation_fraction",
            "max_imputation_fraction",
        ]:
            _merge_node_column(out, bias_rows, column)

    enrichment = aux["enrichment"]
    if not enrichment.empty:
        enrich_rows = enrichment[enrichment["config_id"].astype(str) == config_id].copy()
        _merge_node_column(out, enrich_rows, "enrichment_z", "discoverymethod_enrichment_z")
        _merge_node_column(out, enrich_rows, "empirical_p_value", "discoverymethod_enrichment_p")
        _merge_node_column(out, enrich_rows, "observed_dominant_method_fraction", "dominant_discoverymethod_fraction")
        _merge_node_column(out, enrich_rows, "dominant_discoverymethod")

    out["final_label"] = out.get("final_label", pd.Series(index=out.index, dtype="object")).fillna("unclassified")
    out["dominant_discoverymethod"] = (
        out.get("dominant_discoverymethod", pd.Series(index=out.index, dtype="object"))
        .combine_first(out.get("discoverymethod_top", pd.Series(index=out.index, dtype="object")))
        .fillna("unknown")
    )
    out["dominant_discoverymethod_fraction"] = pd.to_numeric(
        out.get("dominant_discoverymethod_fraction"), errors="coerce"
    ).combine_first(pd.Series(np.nan, index=out.index))
    out["mean_imputation_fraction"] = pd.to_numeric(out.get("mean_imputation_fraction"), errors="coerce").fillna(0.0)
    out["n_members"] = pd.to_numeric(out.get("n_members"), errors="coerce").fillna(1)
    out["discoverymethod_enrichment_z"] = pd.to_numeric(
        out.get("discoverymethod_enrichment_z"), errors="coerce"
    )
    out["discoverymethod_enrichment_p"] = pd.to_numeric(
        out.get("discoverymethod_enrichment_p"), errors="coerce"
    )

    out["candidate_population_plot"] = (
        out.get("candidate_population_top", pd.Series(index=out.index, dtype="object"))
        .combine_first(out.get("dominant_candidate_population", pd.Series(index=out.index, dtype="object")))
        .fillna("unknown_mixed")
    )
    out["radius_class_plot"] = (
        out.get("radius_class_top", pd.Series(index=out.index, dtype="object"))
        .combine_first(out.get("radius_class_dominant", pd.Series(index=out.index, dtype="object")))
        .fillna("unknown")
    )
    out["orbit_class_plot"] = (
        out.get("orbit_class_top", pd.Series(index=out.index, dtype="object"))
        .combine_first(out.get("orbit_class_dominant", pd.Series(index=out.index, dtype="object")))
        .fillna("unknown")
    )
    out["thermal_class_plot"] = (
        out.get("thermal_class_top", pd.Series(index=out.index, dtype="object"))
        .combine_first(out.get("thermal_class_dominant", pd.Series(index=out.index, dtype="object")))
        .fillna("unknown")
    )
    return out


def build_graph(config_id: str, nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for node_id in nodes["node_id"].astype(str):
        graph.add_node(node_id)
    if not edges.empty and {"source", "target"}.issubset(edges.columns):
        for row in edges.itertuples(index=False):
            source = str(getattr(row, "source"))
            target = str(getattr(row, "target"))
            if source in graph and target in graph:
                graph.add_edge(source, target)
    return graph


def normalize_positions(raw_positions: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    if not raw_positions:
        return {}
    ids = list(raw_positions)
    xy = np.array([raw_positions[node_id] for node_id in ids], dtype=float)
    xy = np.nan_to_num(xy, nan=0.0)
    xy[:, 0] -= np.nanmean(xy[:, 0])
    xy[:, 1] -= np.nanmean(xy[:, 1])
    scale = max(float(np.nanmax(np.abs(xy))), 1e-9)
    xy = xy / scale
    return {node_id: (float(point[0]), float(point[1])) for node_id, point in zip(ids, xy)}


def compute_positions(graph: nx.Graph, nodes: pd.DataFrame, layout: str) -> tuple[dict[str, tuple[float, float]], str]:
    x_candidates = ["x", "pos_x", "layout_x", "lens_1_mean"]
    y_candidates = ["y", "pos_y", "layout_y", "lens_2_mean"]
    if layout in {"auto", "lens"}:
        for x_col in x_candidates:
            for y_col in y_candidates:
                if x_col in nodes.columns and y_col in nodes.columns:
                    coords = nodes[["node_id", x_col, y_col]].copy()
                    coords[x_col] = pd.to_numeric(coords[x_col], errors="coerce")
                    coords[y_col] = pd.to_numeric(coords[y_col], errors="coerce")
                    if coords[[x_col, y_col]].notna().all(axis=None):
                        raw = {
                            str(row.node_id): (float(getattr(row, x_col)), float(getattr(row, y_col)))
                            for row in coords.itertuples(index=False)
                        }
                        return normalize_positions(raw), f"{x_col}/{y_col}"
        if layout == "lens":
            raise ValueError("Requested lens layout, but no complete x/y or lens coordinate columns were found.")

    if layout == "kamada":
        return nx.kamada_kawai_layout(graph), "networkx.kamada_kawai_layout"

    k_value = 1.1 / math.sqrt(max(graph.number_of_nodes(), 1))
    return nx.spring_layout(graph, seed=42, k=k_value, iterations=200), "networkx.spring_layout(seed=42)"


def load_graph_render_data(config_id: str, aux: dict[str, pd.DataFrame], layout: str = "auto") -> GraphRenderData:
    g_path = graph_path(config_id)
    n_path = node_path(config_id)
    e_path = edge_path(config_id)
    if not g_path.exists():
        raise FileNotFoundError(f"Missing graph JSON: {g_path}")
    if not n_path.exists():
        raise FileNotFoundError(f"Missing node CSV: {n_path}")
    if not e_path.exists():
        raise FileNotFoundError(f"Missing edge CSV: {e_path}")

    # The JSON is loaded to validate the graph artifact and keep the script tied to existing Mapper outputs.
    load_json(g_path)
    nodes = enrich_nodes(config_id, read_csv(n_path), aux)
    edges = read_csv(e_path, required=False)
    graph = build_graph(config_id, nodes, edges)
    positions, layout_used = compute_positions(graph, nodes, layout)
    return GraphRenderData(
        config_id=config_id,
        title=SHORT_TITLES.get(config_id, config_id),
        graph=graph,
        nodes=nodes,
        positions=positions,
        layout_used=layout_used,
        graph_path=g_path,
        node_path=n_path,
        edge_path=e_path,
    )


def node_sizes(nodes: pd.DataFrame, min_size: float = 20.0, max_size: float = 220.0) -> np.ndarray:
    counts = np.sqrt(pd.to_numeric(nodes["n_members"], errors="coerce").fillna(1).clip(lower=1).to_numpy())
    if np.nanmax(counts) - np.nanmin(counts) < 1e-9:
        return np.full(len(counts), (min_size + max_size) / 2.0)
    scaled = (counts - np.nanmin(counts)) / (np.nanmax(counts) - np.nanmin(counts))
    return min_size + scaled * (max_size - min_size)


def padded_limits(positions: dict[str, tuple[float, float]], pad: float = 0.12) -> tuple[tuple[float, float], tuple[float, float]]:
    xy = np.array(list(positions.values()), dtype=float)
    if xy.size == 0:
        return (-1, 1), (-1, 1)
    x_min, x_max = float(xy[:, 0].min()), float(xy[:, 0].max())
    y_min, y_max = float(xy[:, 1].min()), float(xy[:, 1].max())
    x_pad = max((x_max - x_min) * pad, 0.05)
    y_pad = max((y_max - y_min) * pad, 0.05)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def ordered_nodes(data: GraphRenderData) -> list[str]:
    return data.nodes["node_id"].astype(str).tolist()


def draw_edges(ax: plt.Axes, data: GraphRenderData) -> None:
    if data.graph.number_of_edges() == 0:
        return
    nx.draw_networkx_edges(
        data.graph,
        data.positions,
        ax=ax,
        edge_color="#A9A9A9",
        alpha=0.5,
        width=0.8,
    )


def draw_categorical_nodes(
    ax: plt.Axes,
    data: GraphRenderData,
    column: str,
    color_map: dict[str, str],
    default: str,
    alpha: float = 0.9,
) -> list[str]:
    node_ids = ordered_nodes(data)
    indexed = data.nodes.set_index("node_id")
    values = indexed.reindex(node_ids)[column].fillna(default).astype(str)
    colors = [color_map.get(value, color_map.get(default, "#BDBDBD")) for value in values]
    xy = np.array([data.positions[node_id] for node_id in node_ids])
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=node_sizes(indexed.reindex(node_ids).reset_index()),
        c=colors,
        alpha=alpha,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )
    return list(dict.fromkeys(values.tolist()))


def draw_numeric_nodes(
    ax: plt.Axes,
    data: GraphRenderData,
    column: str,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> Any:
    node_ids = ordered_nodes(data)
    indexed = data.nodes.set_index("node_id")
    values = pd.to_numeric(indexed.reindex(node_ids)[column], errors="coerce").fillna(0.0).to_numpy()
    xy = np.array([data.positions[node_id] for node_id in node_ids])
    return ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=node_sizes(indexed.reindex(node_ids).reset_index()),
        c=values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.92,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )


def finish_axis(ax: plt.Axes, data: GraphRenderData, title: str | None = None) -> None:
    ax.set_title(title or data.title, fontsize=15, pad=8)
    xlim, ylim = padded_limits(data.positions)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("auto")
    for spine in ax.spines.values():
        spine.set_visible(False)


def label_highlighted_nodes(ax: plt.Axes, data: GraphRenderData, max_labels: int = 10) -> list[str]:
    nodes = data.nodes.copy()
    nodes["node_id"] = nodes["node_id"].astype(str)
    nodes["n_members"] = pd.to_numeric(nodes["n_members"], errors="coerce").fillna(0)
    nodes["discoverymethod_enrichment_z"] = pd.to_numeric(
        nodes["discoverymethod_enrichment_z"], errors="coerce"
    ).fillna(-np.inf)

    label_rows = pd.concat(
        [
            nodes[nodes["final_label"].isin(["physical", "observational"])],
            nodes.sort_values("discoverymethod_enrichment_z", ascending=False).head(5),
        ],
        ignore_index=True,
    ).drop_duplicates("node_id")
    label_rows = label_rows.sort_values(
        ["final_label", "discoverymethod_enrichment_z", "n_members"],
        ascending=[True, False, False],
    ).head(max_labels)

    labels: list[str] = []
    for row in label_rows.itertuples(index=False):
        node_id = str(row.node_id)
        if node_id not in data.positions:
            continue
        x, y = data.positions[node_id]
        text = node_id.replace("_cluster", "\ncl")
        ax.text(
            x,
            y,
            text,
            fontsize=7.5,
            ha="center",
            va="center",
            color="#202020",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            zorder=4,
        )
        labels.append(node_id)
    return labels


def save_figure(fig: plt.Figure, basename: str) -> list[Path]:
    LATEX_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix, kwargs in [
        ("pdf", {"bbox_inches": "tight"}),
        ("png", {"dpi": 300, "bbox_inches": "tight"}),
    ]:
        path = LATEX_FIGURE_DIR / f"{basename}.{suffix}"
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def render_main_graphs_by_evidence(data_by_config: dict[str, GraphRenderData]) -> list[Path]:
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), constrained_layout=False)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.13, wspace=0.22, hspace=0.38)
    fig.suptitle("Selected Mapper graphs by evidence class", fontsize=20, y=0.96)
    for ax, config_id in zip(axes.ravel(), SELECTED_CONFIGS):
        data = data_by_config[config_id]
        draw_edges(ax, data)
        draw_categorical_nodes(ax, data, "final_label", EVIDENCE_COLORS, "unclassified")
        finish_axis(ax, data)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=9, label=label)
        for label, color in EVIDENCE_COLORS.items()
        if label != "unclassified"
    ]
    handles.append(
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=EVIDENCE_COLORS["unclassified"], markersize=9, label="unclassified")
    )
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    return save_figure(fig, "main_graphs_by_evidence_class")


def render_orbital_evidence(data: GraphRenderData) -> list[Path]:
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=False)
    fig.subplots_adjust(left=0.04, right=0.78, top=0.84, bottom=0.08)
    fig.suptitle("Orbital Mapper by evidence class", fontsize=18, y=0.96)
    draw_edges(ax, data)
    draw_categorical_nodes(ax, data, "final_label", EVIDENCE_COLORS, "unclassified")
    label_highlighted_nodes(ax, data)
    finish_axis(ax, data, "Orbital")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=9, label=label)
        for label, color in EVIDENCE_COLORS.items()
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    return save_figure(fig, "orbital_graph_evidence_class")


def render_orbital_discoverymethod(data: GraphRenderData) -> list[Path]:
    methods = data.nodes["dominant_discoverymethod"].fillna("unknown").astype(str)
    dynamic_colors = dict(METHOD_COLORS)
    palette = plt.get_cmap("tab20").colors
    for idx, method in enumerate(sorted(set(methods))):
        dynamic_colors.setdefault(method, matplotlib.colors.to_hex(palette[idx % len(palette)]))

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=False)
    fig.subplots_adjust(left=0.04, right=0.78, top=0.84, bottom=0.08)
    fig.suptitle("Orbital Mapper by dominant discovery method", fontsize=18, y=0.96)
    draw_edges(ax, data)
    used = draw_categorical_nodes(ax, data, "dominant_discoverymethod", dynamic_colors, "unknown")
    label_highlighted_nodes(ax, data)
    finish_axis(ax, data, "Orbital")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=dynamic_colors[method], markersize=9, label=method)
        for method in sorted(used)
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)
    return save_figure(fig, "orbital_graph_discoverymethod")


def render_orbital_imputation(data: GraphRenderData) -> list[Path]:
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=False)
    fig.subplots_adjust(left=0.04, right=0.86, top=0.84, bottom=0.08)
    fig.suptitle("Orbital Mapper by mean imputation fraction", fontsize=18, y=0.96)
    draw_edges(ax, data)
    scatter = draw_numeric_nodes(ax, data, "mean_imputation_fraction", cmap="viridis", vmin=0.0, vmax=1.0)
    label_highlighted_nodes(ax, data)
    finish_axis(ax, data, "Orbital")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Mean imputation fraction", fontsize=10)
    return save_figure(fig, "orbital_graph_imputation")


def render_region_class_counts(aux: dict[str, pd.DataFrame]) -> list[Path]:
    synthesis = aux["synthesis"]
    if synthesis.empty:
        raise FileNotFoundError("final_region_synthesis.csv is required for region_class_counts.")
    counts = (
        synthesis.groupby(["region_type", "final_label"], dropna=False)
        .size()
        .reset_index(name="n_regions")
    )
    pivot = (
        counts.pivot(index="region_type", columns="final_label", values="n_regions")
        .fillna(0)
        .reindex(columns=["physical", "observational", "mixed", "weak"], fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    x = np.arange(len(pivot.index))
    width = 0.18
    for idx, label in enumerate(pivot.columns):
        ax.bar(
            x + (idx - 1.5) * width,
            pivot[label].to_numpy(),
            width=width,
            label=label,
            color=EVIDENCE_COLORS.get(label, "#BDBDBD"),
        )
    ax.set_title("Final region synthesis counts", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(item) for item in pivot.index])
    ax.set_ylabel("Number of regions")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    return save_figure(fig, "region_class_counts")


def display_label(label: str) -> str:
    return str(label).replace("_", " ")


def legend_order(color_map: dict[str, str], values: list[str], default: str) -> list[str]:
    present = set(values)
    ordered = [label for label in color_map if label in present]
    extras = sorted(label for label in present if label not in color_map)
    if default in color_map and default not in ordered:
        ordered.append(default)
    return ordered + extras


def render_main_graphs_by_category(
    data_by_config: dict[str, GraphRenderData],
    column: str,
    color_map: dict[str, str],
    default: str,
    basename: str,
    suptitle: str,
) -> list[Path]:
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), constrained_layout=False)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.14, wspace=0.22, hspace=0.38)
    fig.suptitle(suptitle, fontsize=20, y=0.96)
    all_values: list[str] = []
    for ax, config_id in zip(axes.ravel(), SELECTED_CONFIGS):
        data = data_by_config[config_id]
        if column not in data.nodes.columns:
            data.nodes[column] = default
        draw_edges(ax, data)
        used = draw_categorical_nodes(ax, data, column, color_map, default)
        all_values.extend(used)
        finish_axis(ax, data)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map.get(label, "#BDBDBD"),
            markersize=9,
            label=display_label(label),
        )
        for label in legend_order(color_map, all_values, default)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, max(1, len(handles))), frameon=False, bbox_to_anchor=(0.5, -0.01))
    return save_figure(fig, basename)


def render_main_graphs_by_numeric(
    data_by_config: dict[str, GraphRenderData],
    column: str,
    basename: str,
    suptitle: str,
    colorbar_label: str,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> list[Path]:
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), constrained_layout=False)
    fig.subplots_adjust(left=0.03, right=0.91, top=0.86, bottom=0.08, wspace=0.22, hspace=0.38)
    fig.suptitle(suptitle, fontsize=20, y=0.96)
    scatter = None
    for ax, config_id in zip(axes.ravel(), SELECTED_CONFIGS):
        data = data_by_config[config_id]
        if column not in data.nodes.columns:
            data.nodes[column] = 0.0
        draw_edges(ax, data)
        scatter = draw_numeric_nodes(ax, data, column, cmap=cmap, vmin=vmin, vmax=vmax)
        finish_axis(ax, data)
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label(colorbar_label, fontsize=11)
    return save_figure(fig, basename)


def render_single_graph_by_category(
    data: GraphRenderData,
    column: str,
    color_map: dict[str, str],
    default: str,
    basename: str,
    suptitle: str,
    label_nodes: bool = True,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=False)
    fig.subplots_adjust(left=0.04, right=0.77, top=0.84, bottom=0.08)
    fig.suptitle(suptitle, fontsize=18, y=0.96)
    if column not in data.nodes.columns:
        data.nodes[column] = default
    draw_edges(ax, data)
    used = draw_categorical_nodes(ax, data, column, color_map, default)
    if label_nodes:
        label_highlighted_nodes(ax, data)
    finish_axis(ax, data, data.title.replace("\n", " "))
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map.get(label, "#BDBDBD"),
            markersize=9,
            label=display_label(label),
        )
        for label in legend_order(color_map, used, default)
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)
    return save_figure(fig, basename)


def render_astrophysical_interpretation_figures(data_by_config: dict[str, GraphRenderData]) -> list[Path]:
    outputs: list[Path] = []
    outputs.extend(
        render_main_graphs_by_category(
            data_by_config,
            "candidate_population_plot",
            CANDIDATE_POPULATION_COLORS,
            "unknown_mixed",
            "astro_main_graphs_by_candidate_population",
            "Selected Mapper graphs by candidate population",
        )
    )
    outputs.extend(
        render_main_graphs_by_category(
            data_by_config,
            "radius_class_plot",
            RADIUS_COLORS,
            "unknown",
            "astro_main_graphs_by_radius_class",
            "Selected Mapper graphs by radius class",
        )
    )
    outputs.extend(
        render_main_graphs_by_category(
            data_by_config,
            "orbit_class_plot",
            ORBIT_COLORS,
            "unknown",
            "astro_main_graphs_by_orbit_class",
            "Selected Mapper graphs by orbit class",
        )
    )
    outputs.extend(
        render_main_graphs_by_numeric(
            data_by_config,
            "mean_imputation_fraction",
            "astro_main_graphs_by_imputation_fraction",
            "Selected Mapper graphs by mean imputation fraction",
            "Mean imputation fraction",
        )
    )
    outputs.extend(
        render_single_graph_by_category(
            data_by_config["orbital_pca2_cubes10_overlap0p35"],
            "orbit_class_plot",
            ORBIT_COLORS,
            "unknown",
            "astro_orbital_graph_by_orbit_class",
            "Orbital Mapper by orbit class",
        )
    )
    outputs.extend(
        render_single_graph_by_category(
            data_by_config["joint_no_density_pca2_cubes10_overlap0p35"],
            "candidate_population_plot",
            CANDIDATE_POPULATION_COLORS,
            "unknown_mixed",
            "astro_joint_no_density_graph_by_candidate_population",
            "Joint no-density Mapper by candidate population",
        )
    )
    outputs.extend(
        render_single_graph_by_category(
            data_by_config["thermal_pca2_cubes10_overlap0p35"],
            "thermal_class_plot",
            THERMAL_COLORS,
            "unknown",
            "astro_thermal_graph_by_thermal_class",
            "Thermal Mapper by thermal class",
        )
    )
    return outputs


def node_hover_text(nodes: pd.DataFrame) -> list[str]:
    hover = []
    for row in nodes.itertuples(index=False):
        get = row._asdict().get
        hover.append(
            "<br>".join(
                [
                    f"node_id: {get('node_id')}",
                    f"n_members: {get('n_members')}",
                    f"final_label: {get('final_label')}",
                    f"dominant_discoverymethod: {get('dominant_discoverymethod')}",
                    f"method_purity: {get('dominant_discoverymethod_fraction')}",
                    f"mean_imputation_fraction: {get('mean_imputation_fraction')}",
                    f"candidate_population: {get('candidate_population_plot')}",
                    f"radius_class: {get('radius_class_plot')}",
                    f"orbit_class: {get('orbit_class_plot')}",
                    f"thermal_class: {get('thermal_class_plot')}",
                ]
            )
        )
    return hover


def plotly_node_sizes(nodes: pd.DataFrame) -> np.ndarray:
    return np.clip(node_sizes(nodes, min_size=7, max_size=26), 7, 26)


def plotly_graph_traces(
    data: GraphRenderData,
    color_values: list[str] | np.ndarray,
    colors: list[str] | np.ndarray,
    hover: list[str],
    showscale: bool = False,
    colorscale: str | None = None,
    cmin: float | None = None,
    cmax: float | None = None,
) -> list[Any]:
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for source, target in data.graph.edges():
        x0, y0 = data.positions[source]
        x1, y1 = data.positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_ids = ordered_nodes(data)
    xy = np.array([data.positions[node_id] for node_id in node_ids])
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={"width": 0.8, "color": "rgba(150,150,150,0.5)"},
        hoverinfo="skip",
        showlegend=False,
    )
    marker: dict[str, Any] = {
        "size": plotly_node_sizes(data.nodes),
        "line": {"width": 0.5, "color": "white"},
    }
    if showscale:
        marker.update(
            {
                "color": color_values,
                "colorscale": colorscale or "Viridis",
                "cmin": cmin,
                "cmax": cmax,
                "showscale": True,
                "colorbar": {"title": "Imputation"},
            }
        )
    else:
        marker["color"] = colors
    node_trace = go.Scatter(
        x=xy[:, 0],
        y=xy[:, 1],
        mode="markers",
        marker=marker,
        text=hover,
        hoverinfo="text",
        showlegend=False,
    )
    return [edge_trace, node_trace]


def plotly_edge_trace(data: GraphRenderData) -> Any:
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for source, target in data.graph.edges():
        x0, y0 = data.positions[source]
        x1, y1 = data.positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    return go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={"width": 0.8, "color": "rgba(150,150,150,0.5)"},
        hoverinfo="skip",
        showlegend=False,
    )


def plotly_category_traces(
    data: GraphRenderData,
    column: str,
    color_map: dict[str, str],
    default: str,
    legend_seen: set[str] | None = None,
) -> list[Any]:
    legend_seen = legend_seen if legend_seen is not None else set()
    if column not in data.nodes.columns:
        data.nodes[column] = default
    work = data.nodes.copy()
    work[column] = work[column].fillna(default).astype(str)
    node_ids = work["node_id"].astype(str).tolist()
    hover = node_hover_text(work)
    hover_by_node = dict(zip(node_ids, hover))
    size_by_node = dict(zip(node_ids, plotly_node_sizes(work)))
    traces = [plotly_edge_trace(data)]
    values = work[column].tolist()
    for label in legend_order(color_map, values, default):
        selected = work[work[column].eq(label)]
        if selected.empty:
            continue
        selected_ids = selected["node_id"].astype(str).tolist()
        xy = np.array([data.positions[node_id] for node_id in selected_ids])
        showlegend = label not in legend_seen
        legend_seen.add(label)
        traces.append(
            go.Scatter(
                x=xy[:, 0],
                y=xy[:, 1],
                mode="markers",
                marker={
                    "size": [size_by_node[node_id] for node_id in selected_ids],
                    "color": color_map.get(label, "#BDBDBD"),
                    "line": {"width": 0.5, "color": "white"},
                },
                text=[hover_by_node[node_id] for node_id in selected_ids],
                hoverinfo="text",
                name=display_label(label),
                legendgroup=label,
                showlegend=showlegend,
            )
        )
    return traces


def plotly_numeric_traces(data: GraphRenderData, column: str, title: str = "Value") -> list[Any]:
    if column not in data.nodes.columns:
        data.nodes[column] = 0.0
    node_ids = ordered_nodes(data)
    xy = np.array([data.positions[node_id] for node_id in node_ids])
    values = pd.to_numeric(data.nodes.set_index("node_id").reindex(node_ids)[column], errors="coerce").fillna(0.0)
    return [
        plotly_edge_trace(data),
        go.Scatter(
            x=xy[:, 0],
            y=xy[:, 1],
            mode="markers",
            marker={
                "size": plotly_node_sizes(data.nodes.set_index("node_id").reindex(node_ids).reset_index()),
                "color": values.to_numpy(),
                "colorscale": "Viridis",
                "cmin": 0.0,
                "cmax": 1.0,
                "showscale": True,
                "colorbar": {"title": title},
                "line": {"width": 0.5, "color": "white"},
            },
            text=node_hover_text(data.nodes.set_index("node_id").reindex(node_ids).reset_index()),
            hoverinfo="text",
            showlegend=False,
        ),
    ]


def write_plotly_category_panel(
    data_by_config: dict[str, GraphRenderData],
    column: str,
    color_map: dict[str, str],
    default: str,
    filename: str,
    title: str,
) -> Path:
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[SHORT_TITLES[config].replace("\n", " ") for config in SELECTED_CONFIGS],
    )
    legend_seen: set[str] = set()
    for idx, config_id in enumerate(SELECTED_CONFIGS):
        data = data_by_config[config_id]
        row = idx // 3 + 1
        col = idx % 3 + 1
        for trace in plotly_category_traces(data, column, color_map, default, legend_seen):
            fig.add_trace(trace, row=row, col=col)
    fig.update_layout(
        title=title,
        height=850,
        width=1450,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.08, "xanchor": "center", "x": 0.5},
        plot_bgcolor="white",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    path = INTERACTIVE_DIR / filename
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def write_plotly_single_category(
    data: GraphRenderData,
    column: str,
    color_map: dict[str, str],
    default: str,
    filename: str,
    title: str,
) -> Path:
    fig = go.Figure()
    legend_seen: set[str] = set()
    for trace in plotly_category_traces(data, column, color_map, default, legend_seen):
        fig.add_trace(trace)
    fig.update_layout(
        title=title,
        height=780,
        width=1050,
        legend={"orientation": "v", "yanchor": "middle", "y": 0.5, "xanchor": "left", "x": 1.02},
        xaxis={"visible": False},
        yaxis={"visible": False},
        plot_bgcolor="white",
        margin={"l": 30, "r": 220, "t": 90, "b": 40},
    )
    path = INTERACTIVE_DIR / filename
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def write_plotly_numeric_panel(
    data_by_config: dict[str, GraphRenderData],
    column: str,
    filename: str,
    title: str,
) -> Path:
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[SHORT_TITLES[config].replace("\n", " ") for config in SELECTED_CONFIGS],
    )
    for idx, config_id in enumerate(SELECTED_CONFIGS):
        data = data_by_config[config_id]
        row = idx // 3 + 1
        col = idx % 3 + 1
        traces = plotly_numeric_traces(data, column, "Imputation")
        # Avoid repeated colorbars in the panel.
        if idx != len(SELECTED_CONFIGS) - 1:
            traces[1].marker.showscale = False
        for trace in traces:
            fig.add_trace(trace, row=row, col=col)
    fig.update_layout(title=title, height=850, width=1450, showlegend=False, plot_bgcolor="white")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    path = INTERACTIVE_DIR / filename
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def write_plotly_astrophysical_html(data_by_config: dict[str, GraphRenderData]) -> list[Path]:
    if go is None or make_subplots is None:
        return []
    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        write_plotly_category_panel(
            data_by_config,
            "candidate_population_plot",
            CANDIDATE_POPULATION_COLORS,
            "unknown_mixed",
            "astro_main_graphs_by_candidate_population.html",
            "Selected Mapper graphs by candidate population",
        ),
        write_plotly_category_panel(
            data_by_config,
            "radius_class_plot",
            RADIUS_COLORS,
            "unknown",
            "astro_main_graphs_by_radius_class.html",
            "Selected Mapper graphs by radius class",
        ),
        write_plotly_category_panel(
            data_by_config,
            "orbit_class_plot",
            ORBIT_COLORS,
            "unknown",
            "astro_main_graphs_by_orbit_class.html",
            "Selected Mapper graphs by orbit class",
        ),
        write_plotly_numeric_panel(
            data_by_config,
            "mean_imputation_fraction",
            "astro_main_graphs_by_imputation_fraction.html",
            "Selected Mapper graphs by mean imputation fraction",
        ),
        write_plotly_single_category(
            data_by_config["orbital_pca2_cubes10_overlap0p35"],
            "orbit_class_plot",
            ORBIT_COLORS,
            "unknown",
            "astro_orbital_graph_by_orbit_class.html",
            "Orbital Mapper by orbit class",
        ),
        write_plotly_single_category(
            data_by_config["joint_no_density_pca2_cubes10_overlap0p35"],
            "candidate_population_plot",
            CANDIDATE_POPULATION_COLORS,
            "unknown_mixed",
            "astro_joint_no_density_graph_by_candidate_population.html",
            "Joint no-density Mapper by candidate population",
        ),
        write_plotly_single_category(
            data_by_config["thermal_pca2_cubes10_overlap0p35"],
            "thermal_class_plot",
            THERMAL_COLORS,
            "unknown",
            "astro_thermal_graph_by_thermal_class.html",
            "Thermal Mapper by thermal class",
        ),
    ]
    return outputs


def write_plotly_html(data_by_config: dict[str, GraphRenderData]) -> list[Path]:
    if go is None or make_subplots is None:
        return []
    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[SHORT_TITLES[config].replace("\n", " ") for config in SELECTED_CONFIGS],
    )
    for idx, config_id in enumerate(SELECTED_CONFIGS):
        data = data_by_config[config_id]
        labels = data.nodes["final_label"].fillna("unclassified").astype(str).tolist()
        colors = [EVIDENCE_COLORS.get(label, EVIDENCE_COLORS["unclassified"]) for label in labels]
        traces = plotly_graph_traces(data, labels, colors, node_hover_text(data.nodes))
        row = idx // 3 + 1
        col = idx % 3 + 1
        for trace in traces:
            fig.add_trace(trace, row=row, col=col)
    fig.update_layout(title="Selected Mapper graphs by evidence class", height=850, width=1400, showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    path = INTERACTIVE_DIR / "main_graphs_by_evidence_class.html"
    fig.write_html(path, include_plotlyjs="cdn")
    outputs.append(path)

    orbital = data_by_config["orbital_pca2_cubes10_overlap0p35"]
    html_specs = [
        (
            "orbital_graph_evidence_class.html",
            "Orbital Mapper by evidence class",
            orbital.nodes["final_label"].fillna("unclassified").astype(str).tolist(),
            [EVIDENCE_COLORS.get(label, EVIDENCE_COLORS["unclassified"]) for label in orbital.nodes["final_label"].fillna("unclassified").astype(str)],
            False,
        ),
        (
            "orbital_graph_discoverymethod.html",
            "Orbital Mapper by dominant discovery method",
            orbital.nodes["dominant_discoverymethod"].fillna("unknown").astype(str).tolist(),
            [METHOD_COLORS.get(label, METHOD_COLORS["unknown"]) for label in orbital.nodes["dominant_discoverymethod"].fillna("unknown").astype(str)],
            False,
        ),
        (
            "orbital_graph_imputation.html",
            "Orbital Mapper by mean imputation fraction",
            pd.to_numeric(orbital.nodes["mean_imputation_fraction"], errors="coerce").fillna(0.0).to_numpy(),
            pd.to_numeric(orbital.nodes["mean_imputation_fraction"], errors="coerce").fillna(0.0).to_numpy(),
            True,
        ),
    ]
    for filename, title, values, colors, numeric in html_specs:
        fig_single = go.Figure()
        for trace in plotly_graph_traces(
            orbital,
            values,
            colors,
            node_hover_text(orbital.nodes),
            showscale=numeric,
            cmin=0.0 if numeric else None,
            cmax=1.0 if numeric else None,
        ):
            fig_single.add_trace(trace)
        fig_single.update_layout(
            title=title,
            height=850,
            width=1000,
            showlegend=False,
            xaxis={"visible": False},
            yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1},
            plot_bgcolor="white",
        )
        path = INTERACTIVE_DIR / filename
        fig_single.write_html(path, include_plotlyjs="cdn")
        outputs.append(path)
    return outputs


def write_manifest(
    data_by_config: dict[str, GraphRenderData],
    static_outputs: list[Path],
    html_outputs: list[Path],
    layout_requested: str,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Graph Render Manifest",
        "",
        "Clean static Mapper figures were generated from existing graph/node/edge/config outputs only.",
        "",
        "## Render Settings",
        "",
        f"- Requested layout: `{layout_requested}`",
        "- Static backend: `matplotlib` + `networkx`",
        "- Node sizing: sqrt membership scaling, capped to 20-220 points for static figures",
        "- Edge rendering: light gray, alpha 0.5, width 0.8, drawn before nodes",
        "- Mapper was not rerun.",
        "- `feature_sets.py` was not modified.",
        "",
        "## Graphs Rendered",
        "",
        "| config_id | title | nodes | edges | layout_used |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for config_id in SELECTED_CONFIGS:
        data = data_by_config[config_id]
        lines.append(
            f"| `{config_id}` | {data.title.replace(chr(10), ' / ')} | "
            f"{data.graph.number_of_nodes()} | {data.graph.number_of_edges()} | `{data.layout_used}` |"
        )
    lines.extend(
        [
            "",
            "## Static Outputs",
            "",
            *[f"- `{path.relative_to(PROJECT_ROOT).as_posix()}`" for path in static_outputs],
            "",
            "## Interactive Outputs",
            "",
        ]
    )
    if html_outputs:
        lines.extend(f"- `{path.relative_to(PROJECT_ROOT).as_posix()}`" for path in html_outputs)
    else:
        lines.append("- Plotly HTML outputs were skipped or Plotly was unavailable.")
    lines.extend(
        [
            "",
            "## Inputs",
            "",
            "- `outputs/mapper/graphs/`",
            "- `outputs/mapper/nodes/`",
            "- `outputs/mapper/edges/`",
            "- `outputs/mapper/tables/final_region_synthesis.csv`",
            "- `outputs/mapper/tables/node_discovery_bias.csv`",
            "- `outputs/mapper/tables/discoverymethod_enrichment_summary.csv`",
            "",
        ]
    )
    MANIFEST_PATH.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render clean static Mapper graph figures from existing artifacts.")
    parser.add_argument(
        "--layout",
        choices=["auto", "lens", "spring", "kamada"],
        default="auto",
        help="Node layout strategy. auto uses existing lens/x-y coordinates when available.",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip optional Plotly interactive HTML outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LATEX_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    aux = load_auxiliary_tables()
    configs = selected_configs()
    missing = [config for config in SELECTED_CONFIGS if config not in configs]
    if missing:
        raise RuntimeError(f"Expected selected pca2 configs missing from main_graph_selection.csv: {missing}")

    data_by_config = {
        config_id: load_graph_render_data(config_id, aux, layout=args.layout)
        for config_id in SELECTED_CONFIGS
    }

    static_outputs: list[Path] = []
    static_outputs.extend(render_main_graphs_by_evidence(data_by_config))
    orbital = data_by_config["orbital_pca2_cubes10_overlap0p35"]
    static_outputs.extend(render_orbital_evidence(orbital))
    static_outputs.extend(render_orbital_discoverymethod(orbital))
    static_outputs.extend(render_orbital_imputation(orbital))
    static_outputs.extend(render_region_class_counts(aux))
    static_outputs.extend(render_astrophysical_interpretation_figures(data_by_config))

    html_outputs: list[Path] = []
    if not args.no_html:
        html_outputs = write_plotly_html(data_by_config)
        html_outputs.extend(write_plotly_astrophysical_html(data_by_config))

    write_manifest(data_by_config, static_outputs, html_outputs, args.layout)
    print("Mapper graph render complete.")
    print(f"Static figures: {LATEX_FIGURE_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Manifest: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    if html_outputs:
        print(f"Interactive HTML: {INTERACTIVE_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
