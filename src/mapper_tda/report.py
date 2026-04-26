from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .metrics import assign_orbital_class, assign_physical_family


WARNING_TEXT = (
    "Mapper es sensible al lens, cover, overlap, clustering, imputacion y escala de variables. "
    "Las estructuras observadas deben considerarse robustas solo si persisten bajo varias cubiertas, "
    "bajo datos completos e imputados, y si no estan dominadas por metodo de descubrimiento."
)

SPACE_LABELS = {
    "phys": "Topologia fisica",
    "orb": "Topologia orbital/energetica",
    "joint": "Topologia conjunta",
}

LENS_LABELS = {
    "pca2": "Lens principal PCA2",
    "density": "Lens de sensibilidad PC1 + densidad local",
    "domain": "Lens interpretativo de dominio",
}

PRESENTATION_METRICS = [
    "n_nodes",
    "n_edges",
    "beta_0",
    "beta_1",
    "graph_density",
    "average_degree",
    "average_clustering",
    "mean_node_size",
]


def _html_table(df: pd.DataFrame, max_rows: int = 40, escape: bool = True) -> str:
    if df is None or df.empty:
        return "<p>No rows.</p>"
    return df.head(max_rows).to_html(index=False, border=0, classes="data-table", escape=escape)


def _link(path: Path | None) -> str:
    if path is None:
        return ""
    return f"<a href='{path.name}'>{path.name}</a>"


def _fig_to_html(fig: go.Figure, bundle_state: dict[str, bool]) -> str:
    include_plotlyjs = "inline" if bundle_state["include_js"] else False
    bundle_state["include_js"] = False
    return fig.to_html(
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        config={
            "displaylogo": False,
            "responsive": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )


def _config_label(space: str, lens: str, n_cubes: int, overlap: float) -> str:
    return f"{space} / {lens} / cubes={n_cubes} / overlap={overlap:.2f}"


def _result_label(result: dict[str, Any]) -> str:
    config = result["config"]
    return _config_label(config.space, config.lens, config.n_cubes, config.overlap)


def _row_label(row: pd.Series) -> str:
    return _config_label(str(row["space"]), str(row["lens"]), int(row["n_cubes"]), float(row["overlap"]))


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _load_numeric_metric_frame(metrics_df: pd.DataFrame) -> pd.DataFrame:
    numeric = pd.DataFrame(index=metrics_df.index)
    for column in PRESENTATION_METRICS:
        if column in metrics_df.columns:
            numeric[column] = _safe_numeric(metrics_df[column])
    return numeric


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    standardized = pd.DataFrame(index=df.index)
    for column in df.columns:
        values = _safe_numeric(df[column])
        mean = values.mean(skipna=True)
        std = values.std(skipna=True, ddof=0)
        if pd.isna(std) or std == 0:
            standardized[column] = values.where(values.isna(), 0.0)
        else:
            standardized[column] = (values - mean) / std
    return standardized


def _highlight_cards(batch_result: dict[str, Any], imputation_context: dict[str, Any]) -> str:
    cards: list[tuple[str, str, str]] = []
    metrics_df = batch_result.get("metrics_df", pd.DataFrame())
    distances_df = batch_result.get("distances_df", pd.DataFrame())

    if not metrics_df.empty:
        max_nodes = metrics_df.sort_values("n_nodes", ascending=False).iloc[0]
        cards.append(
            (
                "Grafo con mas nodos",
                _row_label(max_nodes),
                f"{int(max_nodes['n_nodes'])} nodos y beta_1={int(max_nodes['beta_1'])}",
            )
        )
        densest = metrics_df.sort_values("graph_density", ascending=False).iloc[0]
        cards.append(
            (
                "Mayor densidad topologica",
                _row_label(densest),
                f"densidad={float(densest['graph_density']):.3f}, clustering={float(densest['average_clustering']):.3f}",
            )
        )

    if not distances_df.empty and distances_df["distance_l2"].notna().any():
        closest = distances_df.dropna(subset=["distance_l2"]).sort_values("distance_l2").iloc[0]
        farthest = distances_df.dropna(subset=["distance_l2"]).sort_values("distance_l2", ascending=False).iloc[0]
        cards.append(
            (
                "Pares mas parecidos",
                f"{closest['graph_a']} vs {closest['graph_b']}",
                f"distancia L2 estandarizada = {float(closest['distance_l2']):.3f}",
            )
        )
        cards.append(
            (
                "Pares mas distintos",
                f"{farthest['graph_a']} vs {farthest['graph_b']}",
                f"distancia L2 estandarizada = {float(farthest['distance_l2']):.3f}",
            )
        )

    coverage = imputation_context.get("coverage_knn", pd.DataFrame())
    if not coverage.empty:
        joint = coverage[coverage["feature_group"] == "MAPPER_JOINT_FEATURES"]
        if not joint.empty:
            row = joint.iloc[0]
            cards.append(
                (
                    "Cobertura del espacio conjunto",
                    f"{float(row['before_complete_pct']):.1f}% -> {float(row['after_complete_pct']):.1f}%",
                    "Porcentaje de filas completas antes y despues de la tabla imputada usada para Mapper.",
                )
            )

    method_comparison = imputation_context.get("method_comparison", pd.DataFrame())
    if not method_comparison.empty and "mean_mae_rank" in method_comparison.columns:
        best = method_comparison.sort_values("mean_mae_rank").iloc[0]
        cards.append(
            (
                "Mejor metodo de imputacion segun rank MAE",
                str(best["method"]),
                f"mean_mae_rank={float(best['mean_mae_rank']):.2f}",
            )
        )

    if not cards:
        return "<p>No highlights available.</p>"

    html_cards = []
    for title, value, detail in cards[:6]:
        html_cards.append(
            "<div class='insight-card'>"
            f"<span>{title}</span>"
            f"<strong>{value}</strong>"
            f"<p>{detail}</p>"
            "</div>"
        )
    return "<div class='insight-grid'>" + "".join(html_cards) + "</div>"


def _build_coverage_chart(coverage_df: pd.DataFrame) -> go.Figure | None:
    if coverage_df.empty:
        return None
    plot_df = coverage_df.copy()
    plot_df["feature_group"] = plot_df["feature_group"].astype(str)
    melted = plot_df.melt(
        id_vars=["feature_group", "method"],
        value_vars=["before_complete_pct", "after_complete_pct"],
        var_name="stage",
        value_name="complete_pct",
    )
    melted["stage"] = melted["stage"].map(
        {
            "before_complete_pct": "Antes",
            "after_complete_pct": "Despues",
        }
    )
    fig = px.bar(
        melted,
        x="feature_group",
        y="complete_pct",
        color="stage",
        barmode="group",
        text="complete_pct",
        color_discrete_sequence=["#f28e2b", "#2f6df6"],
        title="Cobertura completa por grupo de features",
    )
    fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
    fig.update_layout(
        xaxis_title="Grupo de features",
        yaxis_title="% de filas completas",
        legend_title="Etapa",
        margin=dict(t=70, l=40, r=20, b=80),
    )
    return fig


def _build_missingness_chart(missingness_df: pd.DataFrame) -> go.Figure | None:
    if missingness_df.empty:
        return None
    plot_df = missingness_df.copy()
    plot_df = plot_df.sort_values("missing_raw_pct", ascending=True)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_df["missing_raw_pct"],
            y=plot_df["feature"],
            name="Missing raw",
            orientation="h",
            marker_color="#d95f5f",
        )
    )
    fig.add_trace(
        go.Bar(
            x=plot_df["missing_after_physical_derivation_pct"],
            y=plot_df["feature"],
            name="Missing tras derivacion fisica",
            orientation="h",
            marker_color="#3a7ca5",
        )
    )
    fig.update_layout(
        barmode="overlay",
        title="Missingness por feature antes de imputar",
        xaxis_title="% de missing",
        yaxis_title="Feature",
        margin=dict(t=70, l=70, r=20, b=50),
    )
    return fig


def _build_validation_heatmap(validation_df: pd.DataFrame) -> go.Figure | None:
    if validation_df.empty:
        return None
    if not {"method", "feature", "spearman"}.issubset(validation_df.columns):
        return None
    pivot = (
        validation_df.pivot_table(index="feature", columns="method", values="spearman", aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )
    if pivot.empty:
        return None
    fig = go.Figure(
        data=
        [
            go.Heatmap(
                z=pivot.to_numpy(dtype=float),
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale="RdYlBu",
                zmin=-1,
                zmax=1,
                colorbar_title="Spearman",
            )
        ]
    )
    fig.update_layout(
        title="Correlacion Spearman en validacion enmascarada",
        xaxis_title="Metodo",
        yaxis_title="Feature",
        margin=dict(t=70, l=80, r=20, b=40),
    )
    return fig


def _build_method_rank_chart(method_comparison: pd.DataFrame) -> go.Figure | None:
    if method_comparison.empty:
        return None
    plot_df = method_comparison.copy()
    value_columns = [column for column in ["mean_mae_rank", "mean_rmse_rank"] if column in plot_df.columns]
    if not value_columns:
        return None
    melted = plot_df.melt(id_vars=["method"], value_vars=value_columns, var_name="metric", value_name="value")
    melted["metric"] = melted["metric"].map(
        {
            "mean_mae_rank": "MAE rank promedio",
            "mean_rmse_rank": "RMSE rank promedio",
        }
    )
    fig = px.bar(
        melted,
        x="method",
        y="value",
        color="metric",
        barmode="group",
        text="value",
        color_discrete_sequence=["#264653", "#e76f51"],
        title="Comparacion de metodos de imputacion",
    )
    fig.update_traces(texttemplate="%{y:.2f}", textposition="outside")
    fig.update_layout(
        xaxis_title="Metodo",
        yaxis_title="Rank promedio (menor es mejor)",
        legend_title="Metrica",
        margin=dict(t=70, l=40, r=20, b=40),
    )
    return fig


def _build_metrics_bar_chart(metrics_df: pd.DataFrame) -> go.Figure | None:
    if metrics_df.empty:
        return None
    plot_df = metrics_df.copy()
    plot_df["config_label"] = plot_df.apply(_row_label, axis=1)
    melted = plot_df.melt(
        id_vars=["config_label", "space", "lens"],
        value_vars=[column for column in ["n_nodes", "n_edges", "beta_1"] if column in plot_df.columns],
        var_name="metric",
        value_name="value",
    )
    if melted.empty:
        return None
    fig = px.bar(
        melted,
        x="config_label",
        y="value",
        color="metric",
        barmode="group",
        title="Tamano y complejidad de cada grafo Mapper",
        color_discrete_sequence=["#1d3557", "#457b9d", "#e76f51"],
    )
    fig.update_layout(
        xaxis_title="Configuracion",
        yaxis_title="Valor",
        margin=dict(t=70, l=40, r=20, b=120),
        legend_title="Metrica",
    )
    return fig


def _build_metrics_heatmap(metrics_df: pd.DataFrame) -> go.Figure | None:
    if metrics_df.empty:
        return None
    numeric = _load_numeric_metric_frame(metrics_df)
    if numeric.empty:
        return None
    standardized = _standardize_columns(numeric).fillna(0.0)
    labels = metrics_df.apply(_row_label, axis=1).tolist()
    fig = go.Figure(
        data=
        [
            go.Heatmap(
                z=standardized.to_numpy(dtype=float),
                x=standardized.columns.tolist(),
                y=labels,
                colorscale="Tealrose",
                zmid=0,
                colorbar_title="z-score",
            )
        ]
    )
    fig.update_layout(
        title="Mapa de metricas topologicas estandarizadas",
        xaxis_title="Metrica",
        yaxis_title="Configuracion",
        margin=dict(t=70, l=120, r=20, b=50),
    )
    return fig


def _build_topology_scatter(metrics_df: pd.DataFrame) -> go.Figure | None:
    if metrics_df.empty:
        return None
    required = {"n_nodes", "beta_1", "mean_node_size", "space", "lens"}
    if not required.issubset(metrics_df.columns):
        return None
    plot_df = metrics_df.copy()
    plot_df["config_label"] = plot_df.apply(_row_label, axis=1)
    fig = px.scatter(
        plot_df,
        x="n_nodes",
        y="beta_1",
        size="mean_node_size",
        color="space",
        symbol="lens",
        hover_name="config_label",
        title="Comparacion global: nodos vs ciclos (beta_1)",
        color_discrete_sequence=["#0f4c5c", "#e36414", "#6a4c93"],
    )
    fig.update_layout(
        xaxis_title="Numero de nodos",
        yaxis_title="Beta_1",
        margin=dict(t=70, l=40, r=20, b=40),
    )
    return fig


def _build_distance_heatmap(distances_df: pd.DataFrame) -> go.Figure | None:
    if distances_df.empty or "distance_l2" not in distances_df.columns:
        return None
    clean = distances_df.dropna(subset=["distance_l2"]).copy()
    if clean.empty:
        return None
    labels = sorted(set(clean["graph_a"]).union(clean["graph_b"]))
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    for label in labels:
        matrix.loc[label, label] = 0.0
    for _, row in clean.iterrows():
        matrix.loc[row["graph_a"], row["graph_b"]] = float(row["distance_l2"])
        matrix.loc[row["graph_b"], row["graph_a"]] = float(row["distance_l2"])
    fig = go.Figure(
        data=
        [
            go.Heatmap(
                z=matrix.to_numpy(dtype=float),
                x=matrix.columns.tolist(),
                y=matrix.index.tolist(),
                colorscale="YlGnBu",
                colorbar_title="Distancia L2",
            )
        ]
    )
    fig.update_layout(
        title="Distancias entre grafos Mapper",
        xaxis_title="Grafo",
        yaxis_title="Grafo",
        margin=dict(t=70, l=120, r=20, b=80),
    )
    return fig


def _network_preview(result: dict[str, Any]) -> go.Figure:
    nx_graph = result["nx_graph"]
    node_table = result["node_table"].set_index("node_id") if not result["node_table"].empty else pd.DataFrame()
    if nx_graph.number_of_nodes() == 0:
        return go.Figure().update_layout(title="Preview de red: grafo vacio")

    positions = result["nx_graph"].copy()
    coords = {}
    try:
        import networkx as nx

        coords = nx.spring_layout(positions, seed=42, k=None)
    except Exception:
        for index, node_id in enumerate(nx_graph.nodes):
            coords[node_id] = (float(index), 0.0)

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for source, target in nx_graph.edges():
        x0, y0 = coords[source]
        x1, y1 = coords[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x: list[float] = []
    node_y: list[float] = []
    node_size: list[float] = []
    node_color: list[float] = []
    hover_text: list[str] = []
    for node_id in nx_graph.nodes():
        x, y = coords[node_id]
        node_x.append(x)
        node_y.append(y)
        if node_id in node_table.index:
            row = node_table.loc[node_id]
            node_size.append(max(14.0, float(row.get("n_points", 1)) * 1.8))
            color_value = row.get("imputed_missing_fraction_mean", 0.0)
            node_color.append(float(color_value) if pd.notna(color_value) else 0.0)
            hover_text.append(
                "<br>".join(
                    [
                        f"node_id={node_id}",
                        f"n_points={int(row.get('n_points', 0))}",
                        f"physical={row.get('physical_family_dominant', 'unknown')}",
                        f"orbital={row.get('orbital_class_dominant', 'unknown')}",
                    ]
                )
            )
        else:
            node_size.append(14.0)
            node_color.append(0.0)
            hover_text.append(f"node_id={node_id}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="rgba(90, 110, 140, 0.35)", width=1.1),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale="Solar",
                line=dict(color="#f8fafc", width=1.2),
                colorbar=dict(title="Frac. imputada"),
            ),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Preview de estructura de red",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(t=60, l=10, r=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def _lens_scatter(result: dict[str, Any]) -> go.Figure:
    work_df = result["work_df"].copy()
    lens = np.asarray(result["lens"])
    work_df["lens_x"] = lens[:, 0]
    work_df["lens_y"] = lens[:, 1]
    work_df["physical_family"] = work_df.apply(assign_physical_family, axis=1)
    work_df["orbital_class"] = work_df.apply(assign_orbital_class, axis=1)

    missing_columns = [column for column in work_df.columns if column.endswith("_was_missing")]
    if missing_columns:
        work_df["row_missing_fraction"] = work_df.loc[:, missing_columns].astype(float).mean(axis=1)
    else:
        work_df["row_missing_fraction"] = 0.0

    hover_data: dict[str, Any] = {
        "physical_family": True,
        "orbital_class": True,
        "row_missing_fraction": ":.2f",
    }
    if "hostname" in work_df.columns:
        hover_data["hostname"] = True

    color_column = result.get("color_column")
    if color_column and color_column in work_df.columns:
        values = _safe_numeric(work_df[color_column])
        if values.notna().any():
            work_df[color_column] = values.fillna(float(values.median()))
            fig = px.scatter(
                work_df,
                x="lens_x",
                y="lens_y",
                color=color_column,
                hover_name="pl_name" if "pl_name" in work_df.columns else None,
                hover_data=hover_data,
                color_continuous_scale="Turbo",
                opacity=0.75,
                title=f"Lens space coloreado por {color_column}",
            )
            fig.update_layout(xaxis_title="Lens 1", yaxis_title="Lens 2", margin=dict(t=60, l=40, r=20, b=40))
            return fig

    fig = px.scatter(
        work_df,
        x="lens_x",
        y="lens_y",
        color="physical_family",
        hover_name="pl_name" if "pl_name" in work_df.columns else None,
        hover_data=hover_data,
        opacity=0.75,
        title="Lens space coloreado por familia fisica",
    )
    fig.update_layout(xaxis_title="Lens 1", yaxis_title="Lens 2", margin=dict(t=60, l=40, r=20, b=40))
    return fig


def _node_size_histogram(result: dict[str, Any]) -> go.Figure:
    node_table = result["node_table"]
    if node_table.empty or "n_points" not in node_table.columns:
        return go.Figure().update_layout(title="Distribucion de tamanos de nodo")
    fig = px.histogram(
        node_table,
        x="n_points",
        nbins=min(18, max(6, len(node_table))),
        title="Distribucion de tamanos de nodo",
        color_discrete_sequence=["#2a9d8f"],
    )
    fig.update_layout(xaxis_title="Puntos por nodo", yaxis_title="Numero de nodos", margin=dict(t=60, l=40, r=20, b=40))
    return fig


def _node_composition_chart(result: dict[str, Any]) -> go.Figure:
    node_table = result["node_table"]
    if node_table.empty:
        return go.Figure().update_layout(title="Composicion dominante por nodos")

    frames: list[pd.DataFrame] = []
    for column, label in [
        ("physical_family_dominant", "physical_family"),
        ("orbital_class_dominant", "orbital_class"),
        ("discoverymethod_dominant", "discoverymethod"),
    ]:
        if column in node_table.columns:
            counts = node_table[column].fillna("unknown").value_counts().rename_axis("label").reset_index(name="count")
            counts["group"] = label
            frames.append(counts)
    if not frames:
        return go.Figure().update_layout(title="Composicion dominante por nodos")

    plot_df = pd.concat(frames, ignore_index=True)
    fig = px.bar(
        plot_df,
        x="label",
        y="count",
        color="group",
        barmode="group",
        title="Etiquetas dominantes por nodo",
        color_discrete_sequence=["#6a4c93", "#ffb703", "#d62828"],
    )
    fig.update_layout(xaxis_title="Etiqueta", yaxis_title="Numero de nodos", margin=dict(t=60, l=40, r=20, b=70))
    return fig


def _top_nodes_table(result: dict[str, Any]) -> pd.DataFrame:
    node_table = result["node_table"].copy()
    if node_table.empty:
        return node_table
    preferred_columns = [
        "node_id",
        "n_points",
        "physical_family_dominant",
        "orbital_class_dominant",
        "discoverymethod_dominant",
        "imputed_missing_fraction_mean",
        "pl_rade_median",
        "pl_bmasse_median",
        "pl_orbper_median",
        "pl_eqt_median",
    ]
    columns = [column for column in preferred_columns if column in node_table.columns]
    return node_table.sort_values("n_points", ascending=False).loc[:, columns].head(8)


def _featured_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not results:
        return []

    lens_order = {"pca2": 0, "density": 1, "domain": 2}

    default_results = [
        result
        for result in results
        if result["config"].n_cubes == 10 and abs(result["config"].overlap - 0.35) < 1e-9
    ]
    if not default_results:
        default_results = results

    ordered = sorted(
        default_results,
        key=lambda result: (
            lens_order.get(result["config"].lens, 99),
            result["config"].space,
            result["config"].n_cubes,
            result["config"].overlap,
        ),
    )

    featured: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for result in ordered:
        pair = (result["config"].space, result["config"].lens)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        featured.append(result)
        if len(featured) >= 6:
            break
    return featured


def _stability_chart(metrics_df: pd.DataFrame, metric: str) -> go.Figure | None:
    if metrics_df.empty or metric not in metrics_df.columns:
        return None
    if not {"space", "lens", "n_cubes", "overlap"}.issubset(metrics_df.columns):
        return None
    plot_df = metrics_df.copy()
    plot_df["facet"] = plot_df["space"].astype(str) + " / " + plot_df["lens"].astype(str)
    if plot_df["facet"].nunique() > 9:
        focus = {"phys / pca2", "orb / pca2", "joint / pca2", "phys / density", "orb / density", "joint / density"}
        plot_df = plot_df[plot_df["facet"].isin(focus)]
    fig = px.line(
        plot_df,
        x="n_cubes",
        y=metric,
        color="overlap",
        facet_col="facet",
        facet_col_wrap=3,
        markers=True,
        title=f"Estabilidad de {metric} bajo la grilla de cubiertas",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(margin=dict(t=80, l=40, r=20, b=40))
    return fig


def _imputation_sections(imputation_context: dict[str, Any], bundle_state: dict[str, bool]) -> list[str]:
    sections: list[str] = []
    coverage_fig = _build_coverage_chart(imputation_context.get("coverage_knn", pd.DataFrame()))
    missingness_fig = _build_missingness_chart(imputation_context.get("missingness_before", pd.DataFrame()))
    validation_fig = _build_validation_heatmap(imputation_context.get("validation_metrics", pd.DataFrame()))
    method_fig = _build_method_rank_chart(imputation_context.get("method_comparison", pd.DataFrame()))

    figure_blocks: list[str] = []
    for title, subtitle, fig in [
        ("Cobertura antes y despues", "Que tan completo queda cada espacio para Mapper.", coverage_fig),
        ("Features con mayor missingness", "Vista previa de donde la imputacion agrega mas soporte.", missingness_fig),
        ("Calidad relativa de imputacion", "Correlacion Spearman por feature y metodo en validacion enmascarada.", validation_fig),
        ("Comparacion entre metodos", "Ranks promedio de error para justificar el baseline operativo.", method_fig),
    ]:
        if fig is None:
            continue
        figure_blocks.append(
            "<div class='chart-card'>"
            f"<h3>{title}</h3>"
            f"<p>{subtitle}</p>"
            f"{_fig_to_html(fig, bundle_state)}"
            "</div>"
        )

    if figure_blocks:
        sections.append(
            "<section>"
            "<div class='section-header'>"
            "<div><p class='eyebrow'>Data readiness</p><h2>Preparacion del dataset e imputacion</h2></div>"
            "<p class='section-copy'>Antes de construir Mapper, este bloque resume la cobertura de features y la calidad relativa de las tablas imputadas usadas como entrada.</p>"
            "</div>"
            "<div class='chart-grid'>"
            + "".join(figure_blocks)
            + "</div></section>"
        )
    return sections


def _global_mapper_sections(batch_result: dict[str, Any], bundle_state: dict[str, bool]) -> list[str]:
    sections: list[str] = []
    metrics_df = batch_result.get("metrics_df", pd.DataFrame())
    distances_df = batch_result.get("distances_df", pd.DataFrame())
    figures = [
        ("Panorama de tamano y complejidad", "Numero de nodos, aristas y ciclos por grafo.", _build_metrics_bar_chart(metrics_df)),
        ("Mapa comparativo de metricas", "Estandarizacion para comparar forma, conectividad y escala entre grafos.", _build_metrics_heatmap(metrics_df)),
        ("Nodos vs ciclos", "Relacion entre tamano del grafo, complejidad y tamano medio de nodo.", _build_topology_scatter(metrics_df)),
        ("Distancias entre grafos", "Que tan parecidas o distantes son las topologias inducidas por cada espacio y lens.", _build_distance_heatmap(distances_df)),
    ]
    chart_blocks: list[str] = []
    for title, subtitle, fig in figures:
        if fig is None:
            continue
        chart_blocks.append(
            "<div class='chart-card'>"
            f"<h3>{title}</h3>"
            f"<p>{subtitle}</p>"
            f"{_fig_to_html(fig, bundle_state)}"
            "</div>"
        )
    if chart_blocks:
        sections.append(
            "<section>"
            "<div class='section-header'>"
            "<div><p class='eyebrow'>Topology comparison</p><h2>Comparacion global de grafos Mapper</h2></div>"
            "<p class='section-copy'>Los graficos principales comparan forma, conectividad, loops y distancia estructural entre espacios fisicos, orbitales y conjuntos.</p>"
            "</div>"
            "<div class='chart-grid'>"
            + "".join(chart_blocks)
            + "</div></section>"
        )
    return sections


def _graph_gallery_sections(batch_result: dict[str, Any], bundle_state: dict[str, bool]) -> list[str]:
    sections: list[str] = []
    results = _featured_results(batch_result.get("results", []))
    if not results:
        return sections

    for result in results:
        config = result["config"]
        metrics = result["graph_metrics"]
        summary_cards = (
            "<div class='mini-metrics'>"
            f"<div class='mini-card'><span>Nodos</span><strong>{int(metrics.get('n_nodes', 0))}</strong></div>"
            f"<div class='mini-card'><span>Aristas</span><strong>{int(metrics.get('n_edges', 0))}</strong></div>"
            f"<div class='mini-card'><span>beta_1</span><strong>{int(metrics.get('beta_1', 0))}</strong></div>"
            f"<div class='mini-card'><span>Rows used</span><strong>{len(result['work_df']):,}</strong></div>"
            "</div>"
        )
        charts = [
            ("Preview de red", "Layout aproximado para leer componentes, hubs y ramas.", _network_preview(result)),
            ("Scatter del lens", "Vista del espacio de cobertura usado por Mapper.", _lens_scatter(result)),
            ("Tamano de nodos", "Distribucion del numero de planetas absorbidos por cada nodo.", _node_size_histogram(result)),
            ("Composicion dominante", "Etiquetas fisicas, orbitales y de discoverymethod que dominan por nodo.", _node_composition_chart(result)),
        ]
        chart_blocks = []
        for title, subtitle, fig in charts:
            chart_blocks.append(
                "<div class='chart-card'>"
                f"<h3>{title}</h3>"
                f"<p>{subtitle}</p>"
                f"{_fig_to_html(fig, bundle_state)}"
                "</div>"
            )
        top_nodes = _top_nodes_table(result)
        sections.append(
            "<section>"
            "<div class='section-header'>"
            f"<div><p class='eyebrow'>{SPACE_LABELS.get(config.space, config.space)}</p>"
            f"<h2>{_result_label(result)}</h2></div>"
            f"<p class='section-copy'>{LENS_LABELS.get(config.lens, config.lens)}. "
            "Esta seccion deja una version leible para presentacion y enlace al HTML interactivo completo.</p>"
            "</div>"
            + summary_cards
            + "<div class='chart-grid'>"
            + "".join(chart_blocks)
            + "</div>"
            + "<div class='table-card'>"
            "<h3>Nodos mas grandes</h3>"
            "<p>Los nodos mas poblados suelen ser buenos puntos de entrada para interpretar agrupamientos robustos.</p>"
            f"{_html_table(top_nodes, 8)}"
            "</div>"
            "</section>"
        )
    return sections


def _stability_sections(batch_result: dict[str, Any], bundle_state: dict[str, bool]) -> list[str]:
    sections: list[str] = []
    stability_df = batch_result.get("stability_grid_df", pd.DataFrame())
    metrics_df = batch_result.get("metrics_df", pd.DataFrame())
    if stability_df is None or stability_df.empty:
        return sections
    figure_blocks = []
    for metric in ["n_nodes", "beta_1"]:
        fig = _stability_chart(metrics_df, metric)
        if fig is None:
            continue
        figure_blocks.append(
            "<div class='chart-card'>"
            f"<h3>Estabilidad de {metric}</h3>"
            "<p>Lectura por lentes y espacios al variar n_cubes y overlap.</p>"
            f"{_fig_to_html(fig, bundle_state)}"
            "</div>"
        )
    sections.append(
        "<section>"
        "<div class='section-header'>"
        "<div><p class='eyebrow'>Robustness</p><h2>Estabilidad bajo la grilla de cubiertas</h2></div>"
        "<p class='section-copy'>Si un patron solo aparece en una esquina del grid, es mas fragil para la presentacion final. Esta vista ayuda a detectarlo rapido.</p>"
        "</div>"
        + ("<div class='chart-grid'>" + "".join(figure_blocks) + "</div>" if figure_blocks else "")
        + "<div class='table-card'>"
        "<h3>Resumen tabular del grid</h3>"
        f"{_html_table(stability_df, 120)}"
        "</div></section>"
    )
    return sections


def build_mapper_report_html(
    batch_result: dict,
    artifact_index: pd.DataFrame,
) -> str:
    metrics_df = batch_result.get("metrics_df", pd.DataFrame())
    distances_df = batch_result.get("distances_df", pd.DataFrame())
    config_rows = batch_result.get("config_summary", {}).get("configs", [])
    dataset_summary = batch_result.get("config_summary", {}).get("dataset", {})
    imputation_context = batch_result.get("imputation_context", {})

    configs_df = pd.DataFrame(config_rows)
    links_df = artifact_index.copy()
    if not links_df.empty:
        for column in ["html_path", "graph_json_path", "node_csv_path", "edge_csv_path", "metrics_json_path"]:
            if column in links_df.columns:
                links_df[column] = links_df[column].apply(lambda value: _link(Path(value)) if value else "")

    bundle_state = {"include_js": True}
    sections: list[str] = []

    sections.append(
        "<section class='hero'>"
        "<div class='hero-copy'>"
        "<p class='eyebrow'>Exoplanet Mapper / TDA</p>"
        "<h1>Reporte visual para comparar topologias fisicas, orbitales y conjuntas</h1>"
        "<p>Este reporte esta pensado para presentacion: resume la preparacion del dataset, "
        "las configuraciones corridas, las metricas principales y una galeria visual por grafo.</p>"
        "</div>"
        "<div class='hero-cards'>"
        f"<div class='hero-card'><span>CSV</span><strong>{dataset_summary.get('csv_name', '')}</strong></div>"
        f"<div class='hero-card'><span>Filas de entrada</span><strong>{dataset_summary.get('rows_input', 0):,}</strong></div>"
        f"<div class='hero-card'><span>Corridas</span><strong>{len(batch_result.get('results', []))}</strong></div>"
        f"<div class='hero-card'><span>Grid</span><strong>{'Si' if dataset_summary.get('grid_mode') else 'No'}</strong></div>"
        "</div>"
        "</section>"
    )

    sections.append(
        "<section>"
        "<div class='section-header'>"
        "<div><p class='eyebrow'>Presentation snapshot</p><h2>Hallazgos descriptivos rapidos</h2></div>"
        "<p class='section-copy'>No son conclusiones cientificas finales: son titulares operativos para orientar la presentacion y decidir que graficos conviene mostrar en vivo.</p>"
        "</div>"
        f"{_highlight_cards(batch_result, imputation_context)}"
        "</section>"
    )

    sections.append(
        "<section class='warning'>"
        f"<strong>Advertencia metodologica.</strong> {WARNING_TEXT}"
        "</section>"
    )

    sections.extend(_imputation_sections(imputation_context, bundle_state))
    sections.extend(_global_mapper_sections(batch_result, bundle_state))
    sections.extend(_graph_gallery_sections(batch_result, bundle_state))
    sections.extend(_stability_sections(batch_result, bundle_state))

    sections.append(
        "<section>"
        "<div class='section-header'>"
        "<div><p class='eyebrow'>Configuration log</p><h2>Configuraciones corridas y artefactos</h2></div>"
        "<p class='section-copy'>Aqui quedan las tablas utiles para respaldo tecnico, anexos o preguntas durante la presentacion.</p>"
        "</div>"
        "<div class='table-card'>"
        "<h3>Configuraciones</h3>"
        f"{_html_table(configs_df, 80)}"
        "</div>"
        "<div class='table-card'>"
        "<h3>Metricas de grafos</h3>"
        f"{_html_table(metrics_df, 80)}"
        "</div>"
        "<div class='table-card'>"
        "<h3>Distancias entre grafos</h3>"
        f"{_html_table(distances_df, 80)}"
        "</div>"
        "<div class='table-card'>"
        "<h3>Artefactos generados</h3>"
        f"{_html_table(links_df, 120, escape=False)}"
        "</div>"
        "</section>"
    )

    css = """
    <style>
      :root {
        --bg: #eef4fb;
        --ink: #132238;
        --muted: #56708a;
        --card: rgba(255, 255, 255, 0.92);
        --line: rgba(31, 62, 104, 0.12);
        --accent: #0f4c5c;
        --accent-2: #ffb703;
        --accent-3: #e76f51;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(30, 136, 229, 0.18), transparent 28%),
          radial-gradient(circle at top right, rgba(255, 183, 3, 0.16), transparent 26%),
          linear-gradient(180deg, #f5f8fc 0%, #ebf2fa 100%);
        font-family: "Aptos", "Segoe UI", "Trebuchet MS", sans-serif;
      }
      main { max-width: 1320px; margin: 0 auto; padding: 28px 22px 64px; }
      section {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 22px;
        margin: 18px 0;
        box-shadow: 0 18px 42px rgba(24, 54, 90, 0.08);
        backdrop-filter: blur(8px);
      }
      .hero {
        display: grid;
        grid-template-columns: 1.35fr 1fr;
        gap: 18px;
        background:
          linear-gradient(135deg, rgba(15, 76, 92, 0.96), rgba(35, 102, 141, 0.88)),
          linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 183, 3, 0.10));
        color: white;
      }
      .hero h1 { margin: 0 0 10px; font-size: 38px; line-height: 1.08; }
      .hero p { margin: 0; color: rgba(255, 255, 255, 0.85); }
      .hero-cards { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
      .hero-card {
        padding: 16px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.18);
      }
      .hero-card span, .insight-card span, .mini-card span { display: block; font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; }
      .hero-card strong { display: block; margin-top: 6px; font-size: 24px; line-height: 1.15; }
      .section-header {
        display: flex;
        justify-content: space-between;
        gap: 18px;
        align-items: flex-start;
        margin-bottom: 16px;
      }
      .section-header h2 { margin: 0; font-size: 28px; line-height: 1.1; }
      .section-copy { max-width: 520px; margin: 0; color: var(--muted); }
      .eyebrow {
        margin: 0 0 8px;
        color: var(--accent);
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-size: 12px;
      }
      .warning {
        background: linear-gradient(90deg, rgba(255, 230, 180, 0.92), rgba(255, 248, 230, 0.92));
        border-color: rgba(190, 140, 40, 0.35);
        font-weight: 600;
      }
      .insight-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
      }
      .insight-card, .table-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 16px;
      }
      .insight-card strong {
        display: block;
        margin: 8px 0 6px;
        font-size: 24px;
        line-height: 1.15;
      }
      .insight-card p { margin: 0; color: var(--muted); }
      .chart-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 16px;
      }
      .chart-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 16px;
      }
      .chart-card h3, .table-card h3 { margin: 0 0 6px; font-size: 18px; }
      .chart-card p, .table-card p { margin: 0 0 12px; color: var(--muted); }
      .mini-metrics {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 14px;
      }
      .mini-card {
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(233, 243, 255, 0.94), rgba(255, 255, 255, 0.96));
        border: 1px solid rgba(31, 62, 104, 0.10);
        padding: 14px 16px;
      }
      .mini-card strong {
        display: block;
        margin-top: 6px;
        font-size: 23px;
      }
      .data-table { border-collapse: collapse; width: 100%; font-size: 13px; background: white; }
      .data-table th, .data-table td {
        border-bottom: 1px solid #e8eef5;
        padding: 8px;
        text-align: left;
        vertical-align: top;
      }
      .data-table th {
        position: sticky;
        top: 0;
        background: #f7fbff;
      }
      a { color: #1a56db; text-decoration: none; }
      a:hover { text-decoration: underline; }
      @media (max-width: 980px) {
        .hero, .section-header { grid-template-columns: 1fr; display: block; }
        .hero-cards, .insight-grid, .chart-grid, .mini-metrics { grid-template-columns: 1fr; }
        .section-copy { max-width: none; margin-top: 12px; }
      }
    </style>
    """

    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>Mapper/TDA presentation report</title>"
        f"{css}</head><body><main>"
        + "".join(sections)
        + "</main></body></html>"
    )
