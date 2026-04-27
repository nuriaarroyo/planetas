from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .feature_sets import SPACE_COMPARISON_ORDER
from .interpretation import build_interpretive_summary_files, generate_interpretation_summary
from .io import ensure_mapper_output_tree, write_json
from .node_selection import MAIN_GRAPH_CONFIGS, build_component_summary, build_highlighted_nodes, build_main_graph_selection
from .planet_classes import add_planet_physical_labels
from .validation import run_bootstrap_validation, run_imputation_method_comparison, run_null_models
from visual_style import LENS_MARKERS, PROJECT_COLOR_CYCLE, SOURCE_PALETTE, apply_axis_style, configure_matplotlib, style_colorbar


def _import_matplotlib():
    import matplotlib

    configure_matplotlib(matplotlib)
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig: Any, pdf_path: Path, png_path: Path | None = None) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    if png_path is not None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, format="png", dpi=180, bbox_inches="tight")


def _message_figure(pdf_path: Path, title: str, message: str, png_path: Path | None = None) -> None:
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.text(0.02, 0.68, message, transform=ax.transAxes, fontsize=11, wrap=True)
    _save_figure(fig, pdf_path, png_path)
    plt.close(fig)


def _config_labels(metrics_df: pd.DataFrame) -> list[str]:
    return [f"{row.space}\n{row.lens}\nc{int(row.n_cubes)} o{row.overlap:.2f}" for row in metrics_df.itertuples(index=False)]


def build_space_comparison(metrics_df: pd.DataFrame) -> pd.DataFrame:
    frame = metrics_df[(metrics_df["lens"] == "pca2") & (metrics_df["n_cubes"] == 10)].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["space"] = pd.Categorical(frame["space"], categories=SPACE_COMPARISON_ORDER, ordered=True)
    return frame.sort_values(["space", "overlap"]).reset_index(drop=True)


def build_lens_sensitivity(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return metrics_df[metrics_df["lens"].isin(["pca2", "density", "domain"])].copy().sort_values(["space", "lens", "n_cubes", "overlap"]).reset_index(drop=True)


def build_density_feature_sensitivity(metrics_df: pd.DataFrame) -> pd.DataFrame:
    pairs = [("phys_min", "phys_density"), ("joint_no_density", "joint")]
    rows: list[dict[str, Any]] = []
    for without_density, with_density in pairs:
        left = metrics_df[(metrics_df["space"] == without_density) & (metrics_df["lens"] == "pca2")]
        right = metrics_df[(metrics_df["space"] == with_density) & (metrics_df["lens"] == "pca2")]
        if left.empty or right.empty:
            continue
        lrow = left.iloc[0]
        rrow = right.iloc[0]
        rows.append(
            {
                "comparison": f"{without_density}_vs_{with_density}",
                "without_density": without_density,
                "with_density": with_density,
                "delta_n_nodes": float(rrow["n_nodes"] - lrow["n_nodes"]),
                "delta_n_edges": float(rrow["n_edges"] - lrow["n_edges"]),
                "delta_beta_1": float(rrow["beta_1"] - lrow["beta_1"]),
                "delta_average_degree": float(rrow["average_degree"] - lrow["average_degree"]),
                "delta_mean_node_imputation_fraction": float(rrow["mean_node_imputation_fraction"] - lrow["mean_node_imputation_fraction"]),
                "delta_mean_node_physically_derived_fraction": float(rrow["mean_node_physically_derived_fraction"] - lrow["mean_node_physically_derived_fraction"]),
            }
        )
    return pd.DataFrame(rows)


def build_imputation_availability(imputation_dir: Path) -> pd.DataFrame:
    methods = ["iterative", "knn", "median", "complete_case"]
    rows = []
    for method in methods:
        rows.append(
            {
                "method": method,
                "mapper_features_exists": (imputation_dir / f"mapper_features_imputed_{method}.csv").exists() if method != "complete_case" else (imputation_dir / "mapper_features_complete_case.csv").exists(),
                "physical_csv_exists": (imputation_dir / f"PSCompPars_imputed_{method}.csv").exists() if method != "complete_case" else False,
            }
        )
    return pd.DataFrame(rows)


def write_primary_artifacts(batch_result: dict[str, Any], outputs_dir: Path) -> dict[str, Path]:
    tree = ensure_mapper_output_tree(outputs_dir)
    paths: dict[str, Path] = {}
    node_tables: list[pd.DataFrame] = []
    edge_tables: list[pd.DataFrame] = []
    planet_labels_frames: list[pd.DataFrame] = []
    for result in batch_result["results"]:
        config = result["config"]
        config_name = result["config_id"]
        graph_path = tree["graphs"] / f"graph_{config_name}.json"
        nodes_path = tree["nodes"] / f"nodes_{config_name}.csv"
        edges_path = tree["edges"] / f"edges_{config_name}.csv"
        config_path = tree["config"] / f"config_{config_name}.json"
        write_json(
            graph_path,
            {
                "config": config.__dict__,
                "mapper_metadata": result["mapper_metadata"],
                "graph_metrics": result["graph_metrics"],
                "graph": {key: value for key, value in result["graph"].items() if key in {"nodes", "links", "simplices", "meta_data", "meta_nodes", "sample_id_lookup"}},
            },
        )
        result["node_table"].to_csv(nodes_path, index=False)
        result["edge_table"].to_csv(edges_path, index=False)
        write_json(config_path, {"config": config.__dict__, "config_id": config_name})
        node_tables.append(result["node_table"])
        edge_tables.append(result["edge_table"])
        labeled = add_planet_physical_labels(result["physical_df"].copy())
        labeled.insert(0, "config_id", config_name)
        planet_labels_frames.append(labeled)

    metrics_path = tree["metrics"] / "mapper_graph_metrics.csv"
    batch_result["metrics_df"].to_csv(metrics_path, index=False)
    distance_path = tree["distances"] / "mapper_graph_distances_metric_l2.csv"
    batch_result["distances_df"].to_csv(distance_path, index=False)
    alignment_path = tree["tables"] / "mapper_input_alignment_summary.csv"
    pd.DataFrame([batch_result["alignment_summary"]]).to_csv(alignment_path, index=False)
    node_audit_path = tree["tables"] / "mapper_node_source_audit.csv"
    pd.concat(node_tables, ignore_index=True).to_csv(node_audit_path, index=False)
    edges_all_path = tree["tables"] / "mapper_edges_all.csv"
    pd.concat(edge_tables, ignore_index=True).to_csv(edges_all_path, index=False)
    labels_path = tree["data"] / "planet_physical_labels.csv"
    pd.concat(planet_labels_frames, ignore_index=True).to_csv(labels_path, index=False)

    paths.update(
        {
            "mapper_graph_metrics": metrics_path,
            "mapper_graph_distances_metric_l2": distance_path,
            "mapper_input_alignment_summary": alignment_path,
            "mapper_node_source_audit": node_audit_path,
            "mapper_edges_all": edges_all_path,
            "planet_physical_labels": labels_path,
        }
    )
    return paths


def write_comparison_tables(batch_result: dict[str, Any], outputs_dir: Path, imputation_outputs_dir: Path) -> dict[str, Path]:
    tree = ensure_mapper_output_tree(outputs_dir)
    metrics_df = batch_result["metrics_df"]
    table_specs = {
        "mapper_space_comparison.csv": build_space_comparison(metrics_df),
        "mapper_lens_sensitivity.csv": build_lens_sensitivity(metrics_df),
        "mapper_density_feature_sensitivity.csv": build_density_feature_sensitivity(metrics_df),
        "mapper_input_availability.csv": build_imputation_availability(imputation_outputs_dir),
    }
    paths: dict[str, Path] = {}
    for filename, frame in table_specs.items():
        path = tree["tables"] / filename
        frame.to_csv(path, index=False)
        paths[path.stem] = path
    return paths


def write_interpretation_tables(batch_result: dict[str, Any], outputs_dir: Path) -> dict[str, Any]:
    tree = ensure_mapper_output_tree(outputs_dir)
    metrics_df = batch_result["metrics_df"]
    main_graph_selection = build_main_graph_selection(metrics_df)
    main_path = tree["tables"] / "main_graph_selection.csv"
    main_graph_selection.to_csv(main_path, index=False)

    principal_results = [result for result in batch_result["results"] if result["config_id"] in MAIN_GRAPH_CONFIGS]
    node_frames: list[pd.DataFrame] = []
    highlight_frames: list[pd.DataFrame] = []
    component_frames: list[pd.DataFrame] = []
    for result in principal_results:
        node_frames.append(result["node_table"].copy())
        highlight = build_highlighted_nodes(result)
        if not highlight.empty:
            highlight_frames.append(highlight)
        component = build_component_summary(result)
        if not component.empty:
            component_frames.append(component)

    node_interp = pd.concat(node_frames, ignore_index=True) if node_frames else pd.DataFrame()
    highlights = pd.concat(highlight_frames, ignore_index=True) if highlight_frames else pd.DataFrame()
    components = pd.concat(component_frames, ignore_index=True) if component_frames else pd.DataFrame()

    node_interp_path = tree["tables"] / "node_physical_interpretation.csv"
    highlighted_path = tree["tables"] / "highlighted_nodes.csv"
    component_path = tree["tables"] / "component_summary.csv"
    node_interp.to_csv(node_interp_path, index=False)
    highlights.to_csv(highlighted_path, index=False)
    components.to_csv(component_path, index=False)

    summary = generate_interpretation_summary(metrics_df, main_graph_selection, highlights, components)
    summary_paths = build_interpretive_summary_files(tree["tables"], summary)

    return {
        "main_graph_selection": main_graph_selection,
        "node_physical_interpretation": node_interp,
        "highlighted_nodes": highlights,
        "component_summary": components,
        "interpretation_summary": summary,
        **summary_paths,
    }


def write_validation_outputs(
    batch_result: dict[str, Any],
    outputs_dir: Path,
    run_bootstrap: bool = False,
    n_bootstrap: int = 30,
    bootstrap_frac: float = 0.8,
    run_null: bool = False,
    n_null: int = 30,
    run_imputation_comparison: bool = False,
) -> dict[str, Any]:
    tree = ensure_mapper_output_tree(outputs_dir)
    payload: dict[str, Any] = {}

    if run_bootstrap:
        bootstrap_metrics, bootstrap_summary = run_bootstrap_validation(batch_result, n_bootstrap=n_bootstrap, bootstrap_frac=bootstrap_frac)
        bootstrap_metrics.to_csv(tree["bootstrap"] / "bootstrap_metrics.csv", index=False)
        bootstrap_summary.to_csv(tree["tables"] / "bootstrap_stability_summary.csv", index=False)
        payload["bootstrap_metrics"] = bootstrap_metrics
        payload["bootstrap_summary"] = bootstrap_summary
    else:
        payload["bootstrap_metrics"] = pd.DataFrame()
        payload["bootstrap_summary"] = pd.DataFrame()

    if run_null:
        null_metrics, null_summary = run_null_models(batch_result, n_null=n_null)
        null_metrics.to_csv(tree["null_models"] / "null_model_metrics.csv", index=False)
        null_summary.to_csv(tree["tables"] / "null_model_summary.csv", index=False)
        payload["null_model_metrics"] = null_metrics
        payload["null_model_summary"] = null_summary
    else:
        payload["null_model_metrics"] = pd.DataFrame()
        payload["null_model_summary"] = pd.DataFrame()

    if run_imputation_comparison:
        compare_df, availability_df = run_imputation_method_comparison(batch_result)
        compare_df.to_csv(tree["tables"] / "imputation_method_mapper_comparison.csv", index=False)
        availability_df.to_csv(tree["tables"] / "mapper_input_availability.csv", index=False)
        payload["imputation_method_mapper_comparison"] = compare_df
        payload["availability_df"] = availability_df
    else:
        payload["imputation_method_mapper_comparison"] = pd.DataFrame()
    return payload


def _bar_complexity(metrics_df: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    if metrics_df.empty:
        _message_figure(pdf_path, "Mapper graph size and complexity", "No data available.", png_path)
        return
    plt = _import_matplotlib()
    labels = _config_labels(metrics_df)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, column, color in zip(axes, ["n_nodes", "n_edges", "beta_1"], PROJECT_COLOR_CYCLE[:3]):
        ax.bar(x, pd.to_numeric(metrics_df[column], errors="coerce").fillna(0), color=color, width=0.72)
        apply_axis_style(ax, ylabel=column)
    apply_axis_style(axes[0], title="Mapper graph size and complexity")
    axes[-1].set_xticks(x, labels, rotation=35, ha="right")
    _save_figure(fig, pdf_path, png_path)
    plt.close(fig)


def _heatmap(frame: pd.DataFrame, columns: list[str], title: str, pdf_path: Path, png_path: Path) -> None:
    if frame.empty:
        _message_figure(pdf_path, title, "No data available.", png_path)
        return
    use_columns = [column for column in columns if column in frame.columns]
    if not use_columns:
        _message_figure(pdf_path, title, "No compatible columns available.", png_path)
        return
    matrix = frame.loc[:, use_columns].apply(pd.to_numeric, errors="coerce")
    matrix = (matrix - matrix.mean()) / matrix.std(ddof=0).replace(0, np.nan)
    matrix = matrix.fillna(0.0)
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(14, max(6, len(frame) * 0.45)))
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="RdYlBu_r")
    apply_axis_style(ax, title=title)
    ax.set_xticks(range(len(use_columns)), use_columns, rotation=35, ha="right")
    ax.set_yticks(range(len(frame)), _config_labels(frame))
    cbar = fig.colorbar(image, ax=ax)
    style_colorbar(cbar, "z-score")
    _save_figure(fig, pdf_path, png_path)
    plt.close(fig)


def _distance_heatmap(distances_df: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    if distances_df.empty or "metric_zscore_l2_distance" not in distances_df.columns:
        _message_figure(pdf_path, "Metric z-score L2 distances", "No pairwise graph distances available.", png_path)
        return
    clean = distances_df.dropna(subset=["metric_zscore_l2_distance"])
    labels = sorted(set(clean["graph_a"]).union(clean["graph_b"]))
    matrix = pd.DataFrame(0.0, index=labels, columns=labels)
    for _, row in clean.iterrows():
        matrix.loc[row["graph_a"], row["graph_b"]] = float(row["metric_zscore_l2_distance"])
        matrix.loc[row["graph_b"], row["graph_a"]] = float(row["metric_zscore_l2_distance"])
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="Blues")
    apply_axis_style(ax, title="Metric z-score L2 distances")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    cbar = fig.colorbar(image, ax=ax)
    style_colorbar(cbar, "metric_zscore_l2_distance")
    _save_figure(fig, pdf_path, png_path)
    plt.close(fig)


def _scatter_nodes_cycles(metrics_df: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    if metrics_df.empty:
        _message_figure(pdf_path, "Nodes vs cycles", "No data available.", png_path)
        return
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {space: color for space, color in zip(sorted(metrics_df["space"].astype(str).unique()), PROJECT_COLOR_CYCLE)}
    for row in metrics_df.itertuples(index=False):
        ax.scatter(
            row.n_nodes,
            row.beta_1,
            color=colors.get(str(row.space), "#334155"),
            marker=LENS_MARKERS.get(str(row.lens), "o"),
            s=92,
            alpha=0.88,
            edgecolors="#ffffff",
            linewidths=0.8,
        )
    apply_axis_style(ax, title="Mapper nodes vs cycles", xlabel="n_nodes", ylabel="beta_1")
    _save_figure(fig, pdf_path, png_path)
    plt.close(fig)


def _simple_bar_from_table(table: pd.DataFrame, title: str, pdf_path: Path, png_path: Path) -> None:
    if table.empty:
        _message_figure(pdf_path, title, "No data available.", png_path)
        return
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 7))
    numeric = table.set_index(table.columns[0]).select_dtypes(include=[np.number])
    if numeric.empty:
        _message_figure(pdf_path, title, "No numeric data available.", png_path)
        plt.close(fig)
        return
    numeric.plot(kind="bar", ax=ax)
    apply_axis_style(ax, title=title, xlabel="")
    ax.tick_params(axis="x", rotation=25)
    _save_figure(fig, pdf_path, png_path)
    plt.close(fig)


def _node_color_values(node_table: pd.DataFrame, column: str) -> pd.Series:
    values_raw = node_table.get(column)
    if values_raw is None:
        return pd.Series(np.zeros(len(node_table)), index=node_table.index, dtype=float)
    if values_raw.dtype == object or str(values_raw.dtype).startswith("string"):
        categories = pd.Categorical(values_raw.astype("string").fillna("unknown"))
        return pd.Series(categories.codes, index=node_table.index, dtype=float)
    return pd.to_numeric(values_raw, errors="coerce").fillna(0.0)


def _graph_network_figure(result: dict[str, Any], color_column: str, title: str, pdf_path: Path, png_path: Path) -> None:
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 8))
    nx_graph = result["nx_graph"]
    if nx_graph.number_of_nodes() == 0:
        _message_figure(pdf_path, title, "Mapper graph vacio para esta configuracion.", png_path)
        plt.close(fig)
        return
    import networkx as nx

    layout = nx.spring_layout(nx_graph, seed=42)
    node_table = result["node_table"].set_index("node_id")
    values = _node_color_values(node_table, color_column)
    nx.draw_networkx_edges(nx_graph, layout, ax=ax, edge_color="#c7d2df", width=1.0, alpha=0.8)
    nodes = nx.draw_networkx_nodes(
        nx_graph,
        layout,
        ax=ax,
        node_color=values.reindex(list(nx_graph.nodes())).fillna(0.0).to_numpy(dtype=float),
        cmap="cividis",
        linewidths=0.9,
        edgecolors="#ffffff",
        node_size=(pd.to_numeric(node_table["n_members"], errors="coerce").reindex(list(nx_graph.nodes())).fillna(1.0) * 16).to_numpy(),
    )
    cbar = fig.colorbar(nodes, ax=ax)
    style_colorbar(cbar, color_column)
    apply_axis_style(ax, title=title)
    ax.axis("off")
    _save_figure(fig, pdf_path, png_path)
    plt.close(fig)


def _lens_scatter_sources(result: dict[str, Any], pdf_path: Path, png_path: Path) -> None:
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 7))
    frame = result["physical_df"].copy()
    lens = np.asarray(result["lens"])
    frame["lens_x"] = lens[:, 0]
    frame["lens_y"] = lens[:, 1]
    source_cols = [column for column in frame.columns if column.endswith("_was_imputed")]
    derived_cols = [column for column in frame.columns if column.endswith("_was_physically_derived")]
    imputed = frame[source_cols].apply(pd.to_numeric, errors="coerce").fillna(0).any(axis=1) if source_cols else pd.Series(False, index=frame.index)
    derived = frame[derived_cols].apply(pd.to_numeric, errors="coerce").fillna(0).any(axis=1) if derived_cols else pd.Series(False, index=frame.index)
    status = np.where(imputed, "imputed", np.where(derived, "physically_derived", "observed"))
    for label in ["observed", "physically_derived", "imputed"]:
        mask = status == label
        ax.scatter(
            frame.loc[mask, "lens_x"],
            frame.loc[mask, "lens_y"],
            s=24,
            alpha=0.74,
            label=label,
            color=SOURCE_PALETTE[label],
            edgecolors="#ffffff",
            linewidths=0.35,
        )
    apply_axis_style(ax, title=pdf_path.stem, xlabel="lens_1", ylabel="lens_2")
    ax.legend()
    _save_figure(fig, pdf_path, png_path)
    plt.close(fig)


def _node_feature_profiles(result: dict[str, Any], pdf_path: Path, png_path: Path) -> None:
    node_table = result["node_table"].copy()
    features = [column for column in node_table.columns if column.startswith("mean_") and column.replace("mean_", "") in result["used_features"]]
    if node_table.empty or not features:
        _message_figure(pdf_path, pdf_path.stem, "No node feature profile data available.", png_path)
        return
    top = node_table.sort_values("n_members", ascending=False).head(8).set_index("node_id")[features]
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 7))
    top.T.plot(ax=ax)
    apply_axis_style(ax, title=pdf_path.stem, xlabel="feature", ylabel="mean physical value")
    _save_figure(fig, pdf_path, png_path)
    plt.close(fig)


def write_figures(batch_result: dict[str, Any], outputs_dir: Path) -> dict[str, Path]:
    tree = ensure_mapper_output_tree(outputs_dir)
    metrics_df = batch_result["metrics_df"]
    distances_df = batch_result["distances_df"]
    density_table = build_density_feature_sensitivity(metrics_df)
    lens_table = build_lens_sensitivity(metrics_df)
    paths: dict[str, Path] = {}
    specs = [
        ("01_mapper_graph_size_complexity.pdf", _bar_complexity, [metrics_df]),
        ("02_mapper_metrics_zscore_heatmap.pdf", _heatmap, [metrics_df, ["n_nodes", "n_edges", "beta_0", "beta_1", "graph_density", "average_degree", "average_clustering", "mean_node_size", "mean_node_imputation_fraction", "mean_node_physically_derived_fraction"], "Mapper metrics z-score heatmap"]),
        ("03_mapper_metric_l2_distances.pdf", _distance_heatmap, [distances_df]),
        ("04_mapper_nodes_vs_cycles.pdf", _scatter_nodes_cycles, [metrics_df]),
        ("05_mapper_imputation_audit_by_config.pdf", _heatmap, [metrics_df, ["mean_node_imputation_fraction", "max_node_imputation_fraction", "frac_nodes_high_imputation"], "Imputation audit by configuration"]),
        ("06_mapper_density_feature_sensitivity.pdf", _simple_bar_from_table, [density_table, "Density feature sensitivity"]),
        ("07_mapper_lens_sensitivity.pdf", _simple_bar_from_table, [lens_table[["space", "lens", "n_nodes"]].pivot_table(index="space", columns="lens", values="n_nodes", aggfunc="mean").reset_index(), "Lens sensitivity by space"] if not lens_table.empty else [pd.DataFrame(), "Lens sensitivity by space"]),
    ]
    for filename, func, args in specs:
        pdf_path = tree["figures_pdf"] / filename
        png_path = tree["figures_png"] / filename.replace(".pdf", ".png")
        func(*args, pdf_path, png_path)
        paths[pdf_path.stem] = pdf_path
    imputation_pdf = tree["figures_pdf"] / "08_mapper_imputation_method_sensitivity.pdf"
    _message_figure(
        imputation_pdf,
        "Imputation method sensitivity",
        "This run did not compute multi-method Mapper comparisons. See mapper_input_availability.csv for available inputs.",
        tree["figures_png"] / "08_mapper_imputation_method_sensitivity.png",
    )
    paths[imputation_pdf.stem] = imputation_pdf

    principal_results = [result for result in batch_result["results"] if result["config"].lens == "pca2"]
    for result in principal_results:
        config_name = result["config_id"]
        for suffix, column in [("network_pl_rade", "mean_pl_rade"), ("network_imputation_fraction", "mean_imputation_fraction"), ("network_physically_derived_fraction", "physically_derived_fraction")]:
            _graph_network_figure(result, column, f"{config_name} {suffix}", tree["figures_pdf"] / f"graph_{config_name}_{suffix}.pdf", tree["figures_png"] / f"graph_{config_name}_{suffix}.png")
        _lens_scatter_sources(result, tree["figures_pdf"] / f"graph_{config_name}_lens_scatter_sources.pdf", tree["figures_png"] / f"graph_{config_name}_lens_scatter_sources.png")
        _node_feature_profiles(result, tree["figures_pdf"] / f"graph_{config_name}_node_feature_profiles.pdf", tree["figures_png"] / f"graph_{config_name}_node_feature_profiles.png")
    return paths


def write_interpretation_figures(
    batch_result: dict[str, Any],
    outputs_dir: Path,
    interpretation_tables: dict[str, Any],
    validation_outputs: dict[str, Any] | None = None,
) -> dict[str, Path]:
    tree = ensure_mapper_output_tree(outputs_dir)
    paths: dict[str, Path] = {}
    principal_results = [result for result in batch_result["results"] if result["config_id"] in MAIN_GRAPH_CONFIGS]
    result_lookup = {result["config_id"]: result for result in principal_results}

    def panel_graphs(column: str, title: str, filename: str) -> None:
        plt = _import_matplotlib()
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        for ax, config_id in zip(axes, MAIN_GRAPH_CONFIGS):
            result = result_lookup.get(config_id)
            if result is None or result["nx_graph"].number_of_nodes() == 0:
                ax.axis("off")
                ax.set_title(config_id)
                continue
            import networkx as nx

            layout = nx.spring_layout(result["nx_graph"], seed=42)
            node_table = result["node_table"].set_index("node_id")
            values_raw = node_table.get(column)
            if values_raw is None:
                values = pd.Series(np.zeros(len(node_table)), index=node_table.index)
            else:
                if values_raw.dtype == object:
                    cat = pd.Categorical(values_raw.astype("string").fillna("unknown"))
                    values = pd.Series(cat.codes, index=node_table.index)
                else:
                    values = pd.to_numeric(values_raw, errors="coerce").fillna(0.0)
            nx.draw_networkx_edges(result["nx_graph"], layout, ax=ax, edge_color="#c6ced8", width=0.8, alpha=0.7)
            nx.draw_networkx_nodes(result["nx_graph"], layout, ax=ax, node_color=values.reindex(list(result["nx_graph"].nodes())).fillna(0.0).to_numpy(dtype=float), cmap="tab20", node_size=(pd.to_numeric(node_table["n_members"], errors="coerce").reindex(list(result["nx_graph"].nodes())).fillna(1.0) * 8).to_numpy())
            ax.set_title(config_id)
            ax.axis("off")
        for ax in axes[len(MAIN_GRAPH_CONFIGS):]:
            ax.axis("off")
        fig.suptitle(title, fontsize=16)
        _save_figure(fig, tree["figures_interpretation_pdf"] / filename, tree["figures_interpretation_png"] / filename.replace(".pdf", ".png"))
        plt.close(fig)
        paths[filename.replace(".pdf", "")] = tree["figures_interpretation_pdf"] / filename

    panel_graphs("candidate_population_top", "Main graphs by heuristic population", "01_main_graphs_by_population.pdf")
    panel_graphs("mean_imputation_fraction", "Main graphs by imputation", "02_main_graphs_by_imputation.pdf")
    panel_graphs("physically_derived_fraction", "Main graphs by physical derivation", "03_main_graphs_by_physical_derivation.pdf")

    orbital = result_lookup.get("orbital_pca2_cubes10_overlap0p35")
    if orbital is not None:
        _graph_network_figure(
            orbital,
            "orbit_class_top",
            "Orbital mapper interpretation",
            tree["figures_interpretation_pdf"] / "04_orbital_mapper_interpretation.pdf",
            tree["figures_interpretation_png"] / "04_orbital_mapper_interpretation.png",
        )
        paths["04_orbital_mapper_interpretation"] = tree["figures_interpretation_pdf"] / "04_orbital_mapper_interpretation.pdf"
    else:
        _message_figure(tree["figures_interpretation_pdf"] / "04_orbital_mapper_interpretation.pdf", "Orbital mapper interpretation", "Orbital graph not available.", tree["figures_interpretation_png"] / "04_orbital_mapper_interpretation.png")

    joint = result_lookup.get("joint_pca2_cubes10_overlap0p35")
    if joint is not None:
        _graph_network_figure(
            joint,
            "candidate_population_top",
            "Joint mapper interpretation",
            tree["figures_interpretation_pdf"] / "05_joint_mapper_interpretation.pdf",
            tree["figures_interpretation_png"] / "05_joint_mapper_interpretation.png",
        )
        paths["05_joint_mapper_interpretation"] = tree["figures_interpretation_pdf"] / "05_joint_mapper_interpretation.pdf"
    else:
        _message_figure(tree["figures_interpretation_pdf"] / "05_joint_mapper_interpretation.pdf", "Joint mapper interpretation", "Joint graph not available.", tree["figures_interpretation_png"] / "05_joint_mapper_interpretation.png")

    density_table = build_density_feature_sensitivity(batch_result["metrics_df"])
    _simple_bar_from_table(density_table, "Density effect node comparison", tree["figures_interpretation_pdf"] / "06_density_effect_node_comparison.pdf", tree["figures_interpretation_png"] / "06_density_effect_node_comparison.png")

    highlights = interpretation_tables.get("highlighted_nodes", pd.DataFrame()).head(15)
    if not highlights.empty:
        _simple_bar_from_table(highlights[["node_id", "n_members", "mean_imputation_fraction"]], "Highlighted nodes table", tree["figures_interpretation_pdf"] / "07_highlighted_nodes_table.pdf", tree["figures_interpretation_png"] / "07_highlighted_nodes_table.png")
    else:
        _message_figure(tree["figures_interpretation_pdf"] / "07_highlighted_nodes_table.pdf", "Highlighted nodes table", "No highlighted nodes available.", tree["figures_interpretation_png"] / "07_highlighted_nodes_table.png")

    components = interpretation_tables.get("component_summary", pd.DataFrame())
    if not components.empty:
        _simple_bar_from_table(components[["component_id", "beta_1_component", "n_members_unique", "mean_imputation_fraction"]], "Component summary", tree["figures_interpretation_pdf"] / "08_component_summary.pdf", tree["figures_interpretation_png"] / "08_component_summary.png")
    else:
        _message_figure(tree["figures_interpretation_pdf"] / "08_component_summary.pdf", "Component summary", "No component summary available.", tree["figures_interpretation_png"] / "08_component_summary.png")

    thermal = result_lookup.get("thermal_pca2_cubes10_overlap0p35")
    if thermal is not None:
        _graph_network_figure(thermal, "mean_imputation_fraction", "Thermal caution", tree["figures_interpretation_pdf"] / "09_thermal_caution.pdf", tree["figures_interpretation_png"] / "09_thermal_caution.png")
    else:
        _message_figure(tree["figures_interpretation_pdf"] / "09_thermal_caution.pdf", "Thermal caution", "Thermal graph not available.", tree["figures_interpretation_png"] / "09_thermal_caution.png")

    bootstrap_summary = (validation_outputs or {}).get("bootstrap_summary", pd.DataFrame())
    if bootstrap_summary is not None and not bootstrap_summary.empty:
        subset = bootstrap_summary[bootstrap_summary["metric"].isin(["beta_1", "n_nodes"])]
        _simple_bar_from_table(subset[["config_id", "mean", "std"]], "Bootstrap stability", tree["figures_interpretation_pdf"] / "10_bootstrap_stability.pdf", tree["figures_interpretation_png"] / "10_bootstrap_stability.png")
    else:
        _message_figure(tree["figures_interpretation_pdf"] / "10_bootstrap_stability.pdf", "Bootstrap stability", "Bootstrap was not run in this execution.", tree["figures_interpretation_png"] / "10_bootstrap_stability.png")

    method_compare = (validation_outputs or {}).get("imputation_method_mapper_comparison", pd.DataFrame())
    if method_compare is not None and not method_compare.empty:
        _simple_bar_from_table(method_compare[["config_id", "n_nodes", "beta_1"]], "Imputation method comparison", tree["figures_interpretation_pdf"] / "11_imputation_method_comparison.pdf", tree["figures_interpretation_png"] / "11_imputation_method_comparison.png")
    else:
        _message_figure(tree["figures_interpretation_pdf"] / "11_imputation_method_comparison.pdf", "Imputation method comparison", "Multi-method Mapper comparison was skipped or partial because not all imputation inputs were available.", tree["figures_interpretation_png"] / "11_imputation_method_comparison.png")

    null_summary = (validation_outputs or {}).get("null_model_summary", pd.DataFrame())
    if null_summary is not None and not null_summary.empty:
        beta_null = null_summary[null_summary["metric"] == "beta_1"]
        _simple_bar_from_table(beta_null[["config_id", "real_value", "null_mean"]], "Null model beta_1", tree["figures_interpretation_pdf"] / "12_null_model_beta1.pdf", tree["figures_interpretation_png"] / "12_null_model_beta1.png")
    else:
        _message_figure(tree["figures_interpretation_pdf"] / "12_null_model_beta1.pdf", "Null model beta_1", "Null models were not run in this execution.", tree["figures_interpretation_png"] / "12_null_model_beta1.png")
    return paths


def write_presentation_figures(batch_result: dict[str, Any], outputs_dir: Path, interpretation_tables: dict[str, Any], validation_outputs: dict[str, Any] | None = None) -> dict[str, Path]:
    tree = ensure_mapper_output_tree(outputs_dir)
    metrics_df = batch_result["metrics_df"]
    summary = interpretation_tables.get("interpretation_summary", generate_interpretation_summary(metrics_df))
    presentation_dir = tree["figures_pdf"] / "presentation"
    paths: dict[str, Path] = {}
    slide_messages = {
        "slide_01_result_summary.pdf": summary.get("global_summary", "No summary available."),
        "slide_03_orbital_signal.pdf": summary.get("imputation_audit", "No orbital signal summary available."),
        "slide_04_density_sensitivity.pdf": summary.get("density_sensitivity", "No density sensitivity summary available."),
        "slide_05_thermal_caution.pdf": "High complexity, high imputation dependence.",
        "slide_07_next_steps.pdf": "bootstrap -> imputation sensitivity -> null models -> scientific node review.",
    }
    for filename, message in slide_messages.items():
        _message_figure(presentation_dir / filename, filename.replace("_", " "), message)
        paths[filename.replace(".pdf", "")] = presentation_dir / filename

    interpretation_dir = tree["figures_interpretation_pdf"]
    copy_specs = {
        "slide_02_main_graphs.pdf": interpretation_dir / "01_main_graphs_by_population.pdf",
        "slide_06_highlighted_nodes.pdf": interpretation_dir / "07_highlighted_nodes_table.pdf",
    }
    for filename, source in copy_specs.items():
        target = presentation_dir / filename
        if source.exists():
            shutil.copyfile(source, target)
        else:
            _message_figure(target, filename.replace("_", " "), "Source figure not available.")
        paths[filename.replace(".pdf", "")] = target
    return paths


def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def _latex_inline_code(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return r"\texttt{" + _latex_escape(match.group(1)) + "}"

    return re.sub(r"`([^`]+)`", repl, text)


def _latexize_summary_text(text: str) -> str:
    return _latex_inline_code(text).replace("beta_1", r"\(\beta_1\)")


def _write_table_tex(frame: pd.DataFrame, path: Path, caption: str, max_rows: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        path.write_text("% No data available.\n", encoding="utf-8")
        return
    latex = frame.head(max_rows).to_latex(
        index=False,
        escape=True,
        longtable=False,
        caption=caption,
        label=f"tab:{path.stem}",
        float_format="%.3f",
        na_rep="NA",
    )
    latex = latex.replace(r"\begin{table}", "\\begin{table}[p]\n\\centering", 1)
    latex = latex.replace(r"\begin{tabular}", "\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}", 1)
    latex = latex.replace(r"\end{tabular}", "\\end{tabular}\n\\end{adjustbox}", 1)
    path.write_text(latex, encoding="utf-8")


def write_latex_report(batch_result: dict[str, Any], outputs_dir: Path, latex_dir: Path, interpretation_tables: dict[str, Any] | None = None, validation_outputs: dict[str, Any] | None = None) -> dict[str, Path]:
    latex_dir.mkdir(parents=True, exist_ok=True)
    sections_dir = latex_dir / "sections"
    figures_dir = latex_dir / "figures"
    tables_dir = latex_dir / "tables"
    for path in [sections_dir, figures_dir, tables_dir]:
        path.mkdir(parents=True, exist_ok=True)
    for stale in sections_dir.glob("*.tex"):
        stale.unlink()
    for stale in tables_dir.glob("*.tex"):
        stale.unlink()

    metrics_df = batch_result["metrics_df"]
    space_table = build_space_comparison(metrics_df)
    density_table = build_density_feature_sensitivity(metrics_df)
    lens_table = build_lens_sensitivity(metrics_df)
    interpretation_tables = interpretation_tables or {}
    validation_outputs = validation_outputs or {}
    summary = interpretation_tables.get(
        "interpretation_summary",
        generate_interpretation_summary(
            metrics_df,
            interpretation_tables.get("main_graph_selection"),
            interpretation_tables.get("highlighted_nodes"),
            interpretation_tables.get("component_summary"),
            validation_outputs.get("bootstrap_summary"),
            validation_outputs.get("null_model_summary"),
            validation_outputs.get("imputation_method_mapper_comparison"),
        ),
    )
    density_text = _latexize_summary_text(summary.get("density_sensitivity", "Not available."))
    lens_text = _latexize_summary_text(summary.get("lens_sensitivity", "Not available."))
    global_summary = _latexize_summary_text(summary.get("global_summary", "No global summary available."))

    for figure in (outputs_dir / "figures_pdf").glob("*.pdf"):
        shutil.copyfile(figure, figures_dir / figure.name)
    for figure in (outputs_dir / "figures_pdf" / "interpretation").glob("*.pdf"):
        shutil.copyfile(figure, figures_dir / figure.name)

    _write_table_tex(metrics_df[["config_id", "n_nodes", "n_edges", "beta_1", "mean_node_imputation_fraction", "mean_node_physically_derived_fraction"]], tables_dir / "mapper_graph_metrics_summary.tex", "Mapper graph metrics summary.")
    _write_table_tex(space_table[["config_id", "n_nodes", "n_edges", "beta_1", "mean_node_imputation_fraction"]], tables_dir / "mapper_space_comparison.tex", "Mapper space comparison.")
    _write_table_tex(density_table, tables_dir / "mapper_density_sensitivity.tex", "Density feature sensitivity.")
    _write_table_tex(lens_table[["config_id", "space", "lens", "n_nodes", "beta_1"]], tables_dir / "mapper_lens_sensitivity.tex", "Lens sensitivity summary.")
    _write_table_tex(metrics_df[["config_id", "mean_node_imputation_fraction", "max_node_imputation_fraction", "frac_nodes_high_imputation"]], tables_dir / "mapper_imputation_audit_summary.tex", "Imputation audit summary.")
    _write_table_tex(interpretation_tables.get("main_graph_selection", pd.DataFrame()), tables_dir / "main_graph_selection_summary.tex", "Main graph selection.")
    _write_table_tex(interpretation_tables.get("highlighted_nodes", pd.DataFrame()), tables_dir / "highlighted_nodes_summary.tex", "Highlighted nodes.")
    _write_table_tex(interpretation_tables.get("component_summary", pd.DataFrame()), tables_dir / "component_summary_short.tex", "Connected component summary.")

    build_interpretive_summary_files(tables_dir, summary)

    key_findings_tex = "\n".join([f"\\item {item}" for item in summary.get("key_findings", [])]) or "\\item not available"
    sections = {
        "00_abstract.tex": (
            "Se construyeron grafos Mapper sobre el catalogo PSCompPars completado. La imputacion principal es iterative. "
            "Los valores se trazan como observed, physically derived o imputed. El objetivo es analisis exploratorio topologico, no prueba final de clases planetarias."
        ),
        "01_introduction.tex": (
            "No interpretamos Mapper como prueba directa de la topologia real de los exoplanetas. Interpretamos los grafos como estructuras inducidas por una matriz completada con trazabilidad explicita. "
            "Las regiones mas confiables son aquellas con baja fraccion de imputacion, estabilidad bajo cambios de lens, imputacion y parametros, y coherencia con variables fisicas planetarias."
        ),
        "02_data_and_imputation.tex": (
            "El dataset usa siete variables principales: pl_rade, pl_bmasse, pl_dens, pl_orbper, pl_orbsmax, pl_insol y pl_eqt. "
            "observed != physically_derived != imputed. "
            "pl_dens es mayoritariamente derivada desde pl_bmasse y pl_rade; por tanto, no debe tratarse como observacion independiente cuando se usa junto con masa y radio."
        ),
        "03_key_findings.tex": "\\begin{itemize}\n" + key_findings_tex + "\n\\end{itemize}\n",
        "04_mapper_methodology.tex": (
            "Sea $X = \\{x_1, \\ldots, x_n\\} \\subset \\mathbb{R}^p$. Sea $f: X \\to \\mathbb{R}^d$ un lens. Para cada cubierta del espacio de lentes se clusteriza la preimagen y se construye un grafo por interseccion de clusters locales. "
            "Usamos $\\beta_1 = E - V + C$ con $C=\\beta_0$. Las etiquetas fisicas de planetas son resumentes heuristicas, no taxonomias confirmadas."
        ),
        "05_mapper_results.tex": (
            f"{global_summary} "
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.95\\linewidth]{figures/01_mapper_graph_size_complexity.pdf}\\caption{Tamano y complejidad de los grafos Mapper.}\\end{figure}"
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.95\\linewidth]{figures/02_mapper_metrics_zscore_heatmap.pdf}\\caption{Heatmap de metricas estandarizadas.}\\end{figure}"
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.75\\linewidth]{figures/04_mapper_nodes_vs_cycles.pdf}\\caption{Relacion entre numero de nodos y ciclos.}\\end{figure}"
        ),
        "06_node_level_interpretation.tex": (
            "These labels are heuristic physical summaries, not confirmed planet taxonomies. "
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.95\\linewidth]{figures/01_main_graphs_by_population.pdf}\\caption{Grafos principales coloreados por poblacion heuristica.}\\end{figure}"
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.95\\linewidth]{figures/02_main_graphs_by_imputation.pdf}\\caption{Grafos principales coloreados por imputacion.}\\end{figure}"
            "\\input{tables/highlighted_nodes_summary.tex}"
        ),
        "07_main_graph_interpretation.tex": (
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.95\\linewidth]{figures/04_orbital_mapper_interpretation.pdf}\\caption{Interpretacion del espacio orbital.}\\end{figure}"
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.95\\linewidth]{figures/05_joint_mapper_interpretation.pdf}\\caption{Interpretacion del espacio conjunto.}\\end{figure}"
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.95\\linewidth]{figures/09_thermal_caution.pdf}\\caption{Complejidad alta pero confianza reducida en el espacio termico.}\\end{figure}"
        ),
        "08_stability_diagnostics.tex": (
            "Bootstrap was not run in this execution." if validation_outputs.get("bootstrap_summary", pd.DataFrame()).empty else
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.95\\linewidth]{figures/10_bootstrap_stability.pdf}\\caption{Bootstrap de estabilidad.}\\end{figure}"
        ),
        "09_null_model_diagnostics.tex": (
            "Null models were not run in this execution." if validation_outputs.get("null_model_summary", pd.DataFrame()).empty else
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.95\\linewidth]{figures/12_null_model_beta1.pdf}\\caption{Diagnostico de modelo nulo para beta_1.}\\end{figure}"
        ),
        "10_sensitivity_analysis.tex": (
            f"{density_text} {lens_text} "
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.9\\linewidth]{figures/06_mapper_density_feature_sensitivity.pdf}\\caption{Sensibilidad al agregar pl\\_dens.}\\end{figure}"
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.9\\linewidth]{figures/07_mapper_lens_sensitivity.pdf}\\caption{Sensibilidad al lens.}\\end{figure}"
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.9\\linewidth]{figures/11_imputation_method_comparison.pdf}\\caption{Comparacion entre metodos de imputacion cuando esta disponible.}\\end{figure}"
        ),
        "11_limitations.tex": (
            "Las limitaciones principales incluyen sesgo observacional por metodo de descubrimiento, variables derivadas no independientes, dependencia de parametros Mapper y el hecho de que un beta_1 alto no implica automaticamente estructura fisica. "
            "La distancia metric_zscore_l2 no es una distancia topologica estricta."
        ),
        "12_conclusion.tex": (
            "El resultado mas prometedor no es una clasificacion final de planetas, sino la identificacion de grafos y regiones candidatos para inspeccion cientifica. "
            "En esta corrida, el espacio orbital PCA es prioritario porque combina estructura no trivial con baja dependencia de imputacion. "
            "El espacio termico es topologicamente complejo, pero debe tratarse con cautela por su alta fraccion de imputacion. "
            "La inclusion de densidad derivada reduce ligeramente la complejidad de Mapper, lo que sugiere que pl_dens actua como coordenada regularizadora mas que como fuente independiente de nueva ramificacion."
        ),
    }
    for filename, body in sections.items():
        (sections_dir / filename).write_text(body + "\n", encoding="utf-8")

    report_body = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{hyperref}
\usepackage{float}
\usepackage{subcaption}
\usepackage{xcolor}
\usepackage{longtable}
\usepackage{array}
\usepackage{placeins}
\usepackage{pdflscape}
\usepackage{adjustbox}
\title{Mapper/TDA Report for Imputed PSCompPars}
\date{}
\begin{document}
\maketitle
\input{sections/00_abstract.tex}
\section{Introduction}
\input{sections/01_introduction.tex}
\section{Data and Imputation}
\input{sections/02_data_and_imputation.tex}
\section{Key Findings}
\input{sections/03_key_findings.tex}
\section{Mapper Methodology}
\input{sections/04_mapper_methodology.tex}
\section{Results}
\input{sections/05_mapper_results.tex}
\section{Node-Level Interpretation}
\input{sections/06_node_level_interpretation.tex}
\section{Main Graph Interpretation}
\input{sections/07_main_graph_interpretation.tex}
\section{Stability Diagnostics}
\input{sections/08_stability_diagnostics.tex}
\section{Null Model Diagnostics}
\input{sections/09_null_model_diagnostics.tex}
\section{Sensitivity Analysis}
\input{sections/10_sensitivity_analysis.tex}
\section{Limitations}
\input{sections/11_limitations.tex}
\section{Conclusion}
\input{sections/12_conclusion.tex}
\FloatBarrier
\clearpage
\begin{landscape}
\footnotesize
\input{tables/mapper_graph_metrics_summary.tex}
\input{tables/mapper_space_comparison.tex}
\input{tables/mapper_density_sensitivity.tex}
\input{tables/mapper_lens_sensitivity.tex}
\input{tables/mapper_imputation_audit_summary.tex}
\input{tables/main_graph_selection_summary.tex}
\input{tables/component_summary_short.tex}
\end{landscape}
\end{document}
"""
    report_path = latex_dir / "mapper_report.tex"
    report_path.write_text(report_body, encoding="utf-8")
    (latex_dir / "README.md").write_text(
        "Compila con `latexmk -pdf -interaction=nonstopmode -halt-on-error mapper_report.tex`.\n",
        encoding="utf-8",
    )
    (latex_dir / "Makefile").write_text(
        "all:\n\tlatexmk -pdf -interaction=nonstopmode -halt-on-error mapper_report.tex\n",
        encoding="utf-8",
    )
    return {"mapper_report_tex": report_path}
