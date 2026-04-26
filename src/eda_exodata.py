from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from feature_config import (
    CLUSTERING_FEATURE_SETS,
    CONTEST_KEY_COLUMNS,
    CONTEST_NUMERIC_COLUMNS,
    DUPLICATE_UNIT_GROUPS,
    IDENTIFIER_COLUMNS,
    LOG_CANDIDATE_COLUMNS,
    NON_MODEL_SUFFIXES,
    RADIUS_BINS,
    RADIUS_LABELS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"


def find_csv(explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"No existe el CSV: {path}")
        return path

    candidates = sorted(PROJECT_ROOT.glob("PSCompPars_*.csv"))
    candidates += sorted((PROJECT_ROOT / "data").glob("PSCompPars_*.csv"))
    candidates += sorted((PROJECT_ROOT / "data" / "raw").glob("PSCompPars_*.csv"))
    if not candidates:
        raise FileNotFoundError("No encontre PSCompPars_*.csv en la raiz, data/ ni data/raw.")
    return candidates[-1]


def parse_column_descriptions(csv_path: Path) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    pattern = re.compile(r"^# COLUMN\s+([^:]+):\s*(.*)$")
    with csv_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            match = pattern.match(line.rstrip("\n"))
            if match:
                descriptions[match.group(1).strip()] = match.group(2).strip()
    return descriptions


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, comment="#", low_memory=False)
    if "pl_dens" not in df.columns and {"pl_bmasse", "pl_rade"}.issubset(df.columns):
        mass = pd.to_numeric(df["pl_bmasse"], errors="coerce")
        radius = pd.to_numeric(df["pl_rade"], errors="coerce")
        valid = (mass > 0) & (radius > 0)
        df["pl_dens"] = np.where(valid, 5.514 * mass / radius.pow(3), np.nan)
        density_source = pd.Series(pd.NA, index=df.index, dtype="object")
        density_source.loc[valid] = "derived_from_pl_bmasse_pl_rade"
        df["pl_dens_source"] = density_source
        df.attrs["derived_columns"] = {
            "pl_dens": "Derived as 5.514 * pl_bmasse / pl_rade^3 because source file did not include pl_dens."
        }
    return df


def column_profile(df: pd.DataFrame, descriptions: dict[str, str]) -> pd.DataFrame:
    rows = []
    total = len(df)
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "column": col,
                "description": descriptions.get(col, ""),
                "dtype": str(s.dtype),
                "non_null": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "missing_pct": round(float(s.isna().mean() * 100), 3),
                "unique": int(s.nunique(dropna=True)),
                "unique_pct": round(float(s.nunique(dropna=True) / total * 100), 3),
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_pct", "column"], ascending=[False, True])


def numeric_profile(df: pd.DataFrame, descriptions: dict[str, str]) -> pd.DataFrame:
    rows = []
    numeric_cols = df.select_dtypes(include="number").columns
    quantiles = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        q = s.quantile(quantiles)
        rows.append(
            {
                "column": col,
                "description": descriptions.get(col, ""),
                "non_null": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "missing_pct": round(float(s.isna().mean() * 100), 3),
                "mean": s.mean(),
                "std": s.std(),
                "min": q.loc[0.0],
                "p01": q.loc[0.01],
                "p05": q.loc[0.05],
                "p25": q.loc[0.25],
                "median": q.loc[0.5],
                "p75": q.loc[0.75],
                "p95": q.loc[0.95],
                "p99": q.loc[0.99],
                "max": q.loc[1.0],
                "skew": s.skew(),
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_pct", "column"], ascending=[False, True])


def key_variable_stats(df: pd.DataFrame, descriptions: dict[str, str]) -> pd.DataFrame:
    available = [col for col in CONTEST_NUMERIC_COLUMNS if col in df.columns]
    prof = numeric_profile(df[available], descriptions)
    prof.insert(1, "role", prof["column"].map(key_roles()))
    return prof


def key_roles() -> dict[str, str]:
    return {
        "pl_rade": "radio planeta",
        "pl_bmasse": "masa planeta",
        "pl_dens": "densidad planeta",
        "pl_orbper": "periodo orbital",
        "pl_orbsmax": "semi eje mayor",
        "pl_orbeccen": "excentricidad",
        "st_teff": "temperatura estrella",
        "st_met": "metalicidad estrella",
        "sy_pnum": "numero planetas",
        "pl_insol": "insolacion",
        "pl_eqt": "temperatura equilibrio",
    }


def clustering_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)
    for name, cols in CLUSTERING_FEATURE_SETS.items():
        available = [col for col in cols if col in df.columns]
        missing_columns = [col for col in cols if col not in df.columns]
        complete = int(df[available].notna().all(axis=1).sum()) if available else 0
        rows.append(
            {
                "feature_set": name,
                "n_features": len(available),
                "missing_features": len(missing_columns),
                "complete_rows": complete,
                "complete_pct": round(complete / total * 100, 3),
                "columns": ", ".join(available),
                "missing_columns": ", ".join(missing_columns),
            }
        )
    return pd.DataFrame(rows).sort_values("complete_pct", ascending=False)


def is_model_auxiliary(col: str) -> bool:
    return col.endswith(NON_MODEL_SUFFIXES) or "_reflink" in col or col.endswith("_systemref")


def candidate_numeric_columns(df: pd.DataFrame, max_missing_pct: float = 35.0) -> list[str]:
    columns: list[str] = []
    for col in df.select_dtypes(include="number").columns:
        if col in IDENTIFIER_COLUMNS:
            continue
        if is_model_auxiliary(col):
            continue
        s = df[col]
        if s.nunique(dropna=True) <= 2:
            continue
        if float(s.isna().mean() * 100) > max_missing_pct:
            continue
        columns.append(col)
    return columns


def strong_correlations(corr: pd.DataFrame, threshold: float = 0.65) -> pd.DataFrame:
    rows = []
    cols = list(corr.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(value) >= threshold:
                rows.append({"feature_a": left, "feature_b": right, "spearman": value})
    return pd.DataFrame(rows).sort_values("spearman", key=lambda s: s.abs(), ascending=False)


def make_missing_fig(profile: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        profile.sort_values("missing_pct", ascending=True),
        x="missing_pct",
        y="column",
        orientation="h",
        hover_data=["description", "non_null", "missing", "unique"],
        title="Porcentaje de valores nulos por columna (320 columnas)",
        labels={"missing_pct": "% nulos", "column": "columna"},
        height=max(900, len(profile) * 12),
    )
    fig.update_layout(margin=dict(l=160, r=30, t=70, b=40))
    return fig


def make_key_missing_fig(profile: pd.DataFrame) -> go.Figure:
    key = profile[profile["column"].isin(CONTEST_KEY_COLUMNS)].copy()
    key["column"] = pd.Categorical(key["column"], CONTEST_KEY_COLUMNS, ordered=True)
    key = key.sort_values("column")
    fig = px.bar(
        key,
        x="column",
        y="missing_pct",
        color="missing_pct",
        color_continuous_scale="Tealrose",
        hover_data=["description", "non_null", "missing", "unique"],
        title="Nulos en variables mencionadas por la guia",
        labels={"missing_pct": "% nulos", "column": "variable"},
        height=520,
    )
    fig.update_layout(xaxis_tickangle=-45, coloraxis_showscale=False)
    return fig


def histogram_trace(values: pd.Series, bins: int = 48) -> tuple[np.ndarray, np.ndarray]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.array([0.0]), np.array([0.0])
    if clean.nunique() == 1:
        val = float(clean.iloc[0])
        return np.array([val]), np.array([len(clean)])
    counts, edges = np.histogram(clean, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, counts


def make_distribution_browser(
    df: pd.DataFrame,
    columns: list[str],
    title: str,
    descriptions: dict[str, str],
    log_positive: bool = False,
) -> go.Figure:
    fig = go.Figure()
    buttons = []
    traces = 0
    for idx, col in enumerate(columns):
        s = pd.to_numeric(df[col], errors="coerce")
        axis_title = col
        if log_positive:
            s = s.where(s > 0).map(lambda value: math.log10(value) if pd.notna(value) else np.nan)
            axis_title = f"log10({col})"
        x, y = histogram_trace(s)
        visible = idx == 0
        fig.add_trace(
            go.Bar(
                x=x,
                y=y,
                visible=visible,
                name=col,
                hovertemplate=f"{html.escape(col)}<br>x=%{{x}}<br>conteo=%{{y}}<extra></extra>",
            )
        )
        visibility = [False] * len(columns)
        visibility[idx] = True
        label = col if len(col) <= 24 else col[:21] + "..."
        desc = descriptions.get(col, "")
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {
                        "xaxis": {"title": axis_title},
                        "title": f"{title}: {col}",
                        "annotations": [
                            {
                                "text": html.escape(desc[:220]),
                                "xref": "paper",
                                "yref": "paper",
                                "x": 0,
                                "y": 1.12,
                                "showarrow": False,
                                "align": "left",
                            }
                        ],
                    },
                ],
            }
        )
        traces += 1

    first = columns[0] if columns else ""
    fig.update_layout(
        title=f"{title}: {first}",
        xaxis_title=f"log10({first})" if log_positive and first else first,
        yaxis_title="conteo",
        height=560,
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 1.0,
                "xanchor": "right",
                "y": 1.18,
                "yanchor": "top",
            }
        ],
        margin=dict(t=120),
    )
    return fig


def make_corr_heatmap(corr: pd.DataFrame, title: str) -> go.Figure:
    fig = px.imshow(
        corr,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        title=title,
        labels={"color": "Spearman"},
        height=720,
    )
    fig.update_xaxes(tickangle=-45)
    return fig


def make_scatter_matrix(df: pd.DataFrame) -> go.Figure:
    work = df.copy()
    selected = ["pl_rade", "pl_bmasse", "pl_orbper", "pl_orbsmax", "st_teff", "st_met"]
    labels = {}
    for col in selected:
        if col in LOG_CANDIDATE_COLUMNS and col in work.columns:
            new_col = f"log10_{col}"
            work[new_col] = pd.to_numeric(work[col], errors="coerce")
            work[new_col] = work[new_col].where(work[new_col] > 0).map(
                lambda value: math.log10(value) if pd.notna(value) else np.nan
            )
            labels[new_col] = f"log10({col})"
        else:
            labels[col] = col

    dimensions = [f"log10_{col}" if col in LOG_CANDIDATE_COLUMNS else col for col in selected]
    dimensions = [col for col in dimensions if col in work.columns]
    plot_df = work[dimensions + ["discoverymethod"]].dropna().copy()
    if len(plot_df) > 2500:
        plot_df = plot_df.sample(2500, random_state=42)
    fig = px.scatter_matrix(
        plot_df,
        dimensions=dimensions,
        color="discoverymethod",
        labels=labels,
        title="Matriz de dispersion para clustering (muestra si hay muchas filas)",
        height=950,
    )
    fig.update_traces(diagonal_visible=False, marker=dict(size=4, opacity=0.55))
    return fig


def make_scatter_figs(df: pd.DataFrame) -> list[go.Figure]:
    figs: list[go.Figure] = []
    if {"pl_orbper", "pl_rade", "discoverymethod", "disc_year"}.issubset(df.columns):
        plot_df = df.dropna(subset=["pl_orbper", "pl_rade"]).copy()
        plot_df = plot_df[(plot_df["pl_orbper"] > 0) & (plot_df["pl_rade"] > 0)]
        fig = px.scatter(
            plot_df,
            x="pl_orbper",
            y="pl_rade",
            color="discoverymethod",
            hover_name="pl_name",
            hover_data=["hostname", "disc_year"],
            log_x=True,
            log_y=True,
            title="Radio vs periodo orbital por metodo de descubrimiento",
            labels={"pl_orbper": "periodo orbital (dias)", "pl_rade": "radio planeta (R_earth)"},
            height=650,
        )
        figs.append(fig)
    if {"pl_bmasse", "pl_rade", "pl_dens", "discoverymethod"}.issubset(df.columns):
        plot_df = df.dropna(subset=["pl_bmasse", "pl_rade", "pl_dens"]).copy()
        plot_df = plot_df[(plot_df["pl_bmasse"] > 0) & (plot_df["pl_rade"] > 0)]
        fig = px.scatter(
            plot_df,
            x="pl_bmasse",
            y="pl_rade",
            color="pl_dens",
            symbol="discoverymethod",
            hover_name="pl_name",
            hover_data=["hostname"],
            log_x=True,
            log_y=True,
            color_continuous_scale="Viridis",
            title="Masa vs radio con color por densidad",
            labels={"pl_bmasse": "masa planeta (M_earth)", "pl_rade": "radio planeta (R_earth)"},
            height=650,
        )
        figs.append(fig)
    if {"pl_insol", "pl_eqt", "pl_rade", "discoverymethod"}.issubset(df.columns):
        plot_df = df.dropna(subset=["pl_insol", "pl_eqt", "pl_rade"]).copy()
        plot_df = plot_df[(plot_df["pl_insol"] > 0) & (plot_df["pl_rade"] > 0)]
        fig = px.scatter(
            plot_df,
            x="pl_insol",
            y="pl_eqt",
            size="pl_rade",
            color="discoverymethod",
            hover_name="pl_name",
            log_x=True,
            title="Insolacion vs temperatura de equilibrio",
            labels={"pl_insol": "insolacion (Tierra=1)", "pl_eqt": "temperatura equilibrio (K)"},
            height=650,
        )
        figs.append(fig)
    return figs


def make_radius_class_fig(df: pd.DataFrame) -> go.Figure:
    work = df.copy()
    work["radius_class"] = pd.cut(
        pd.to_numeric(work["pl_rade"], errors="coerce"),
        bins=RADIUS_BINS,
        labels=RADIUS_LABELS,
    )
    counts = work["radius_class"].value_counts(dropna=False).rename_axis("radius_class").reset_index(name="count")
    counts["radius_class"] = counts["radius_class"].astype(str)
    fig = px.bar(
        counts,
        x="radius_class",
        y="count",
        title="Clasificacion por radio sugerida en la guia",
        labels={"radius_class": "clase por radio", "count": "planetas"},
        height=520,
    )
    fig.update_layout(xaxis_tickangle=-30)
    return fig


def make_discovery_method_fig(df: pd.DataFrame) -> go.Figure:
    counts = df["discoverymethod"].value_counts(dropna=False).rename_axis("method").reset_index(name="count")
    fig = px.bar(
        counts,
        x="count",
        y="method",
        orientation="h",
        title="Planetas por metodo de descubrimiento",
        labels={"count": "planetas", "method": "metodo"},
        height=520,
    )
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    return fig


def make_coverage_fig(coverage: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        coverage,
        x="feature_set",
        y="complete_pct",
        hover_data=["complete_rows", "n_features", "columns"],
        title="Cobertura de conjuntos de variables para clustering",
        labels={"complete_pct": "% filas completas", "feature_set": "conjunto"},
        height=520,
    )
    fig.update_layout(xaxis_tickangle=-25)
    return fig


def dataframe_html(df: pd.DataFrame, max_rows: int = 30) -> str:
    view = df.head(max_rows).copy()
    return view.to_html(index=False, classes="data-table", border=0, escape=False)


def fig_html(fig: go.Figure, include_plotlyjs: bool) -> str:
    return pio.to_html(fig, include_plotlyjs=include_plotlyjs, full_html=False)


def build_report(
    csv_path: Path,
    df: pd.DataFrame,
    profile: pd.DataFrame,
    key_stats: pd.DataFrame,
    numeric_all: pd.DataFrame,
    coverage: pd.DataFrame,
    corr_core: pd.DataFrame,
    corr_candidates: pd.DataFrame,
    strong_corr: pd.DataFrame,
    descriptions: dict[str, str],
) -> str:
    figures: list[go.Figure] = []
    figures.append(make_key_missing_fig(profile))
    figures.append(make_missing_fig(profile))
    figures.append(make_distribution_browser(df, [c for c in CONTEST_NUMERIC_COLUMNS if c in df.columns], "Distribuciones clave", descriptions))
    figures.append(
        make_distribution_browser(
            df,
            [c for c in CONTEST_NUMERIC_COLUMNS if c in LOG_CANDIDATE_COLUMNS and c in df.columns],
            "Distribuciones clave en escala log10",
            descriptions,
            log_positive=True,
        )
    )
    all_numeric_cols = list(df.select_dtypes(include="number").columns)
    figures.append(make_distribution_browser(df, all_numeric_cols, "Navegador de distribuciones numericas", descriptions))
    figures.append(make_corr_heatmap(corr_core, "Correlacion Spearman: variables clave del concurso"))
    if not corr_candidates.empty:
        figures.append(make_corr_heatmap(corr_candidates, "Correlacion Spearman: candidatas numericas con <=35% nulos"))
    figures.append(make_scatter_matrix(df))
    figures.extend(make_scatter_figs(df))
    figures.append(make_radius_class_fig(df))
    figures.append(make_discovery_method_fig(df))
    figures.append(make_coverage_fig(coverage))

    snippets = []
    for idx, fig in enumerate(figures):
        snippets.append(fig_html(fig, include_plotlyjs=(idx == 0)))

    missing_over_80 = profile[profile["missing_pct"] >= 80].shape[0]
    missing_over_50 = profile[profile["missing_pct"] >= 50].shape[0]
    derived_columns = df.attrs.get("derived_columns", {})
    derived_note = ""
    if derived_columns:
        items = "".join(
            f"<li><code>{html.escape(column)}</code>: {html.escape(description)}</li>"
            for column, description in derived_columns.items()
        )
        derived_note = (
            "<section class='note'><h2>Variables derivadas</h2>"
            "<p>Este archivo no traia todas las variables clave del PDF, asi que se agregaron columnas derivadas "
            "solo para el analisis reproducible:</p>"
            f"<ul>{items}</ul></section>"
        )
    key_missing = profile[profile["column"].isin(CONTEST_KEY_COLUMNS)][
        ["column", "description", "missing_pct", "non_null", "unique"]
    ].sort_values("missing_pct", ascending=False)

    css = """
    <style>
      body { font-family: Segoe UI, Arial, sans-serif; margin: 0; color: #1f2933; background: #f7f8fa; }
      main { max-width: 1180px; margin: 0 auto; padding: 28px 24px 56px; }
      section { background: white; border: 1px solid #dde3ea; border-radius: 8px; padding: 20px; margin: 18px 0; }
      h1, h2 { margin: 0 0 12px; }
      p, li { line-height: 1.45; }
      code { background: #eef2f6; padding: 2px 5px; border-radius: 4px; }
      .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
      .metric { background: #f3f7f6; border: 1px solid #d6e2df; border-radius: 8px; padding: 12px; }
      .metric strong { display: block; font-size: 22px; }
      .data-table { border-collapse: collapse; width: 100%; font-size: 13px; }
      .data-table th, .data-table td { border-bottom: 1px solid #e3e8ee; padding: 7px 8px; text-align: left; vertical-align: top; }
      .data-table th { background: #eef2f6; position: sticky; top: 0; }
      .table-wrap { overflow-x: auto; max-height: 620px; }
      .note { background: #fff8df; border-color: #eadb9b; }
    </style>
    """

    summary = {
        "csv": str(csv_path.name),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": int(df.select_dtypes(include="number").shape[1]),
        "missing_over_50_pct_columns": int(missing_over_50),
        "missing_over_80_pct_columns": int(missing_over_80),
    }

    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>ExoData EDA Plotly</title>",
        css,
        "</head><body><main>",
        "<h1>ExoData Challenge - EDA interactivo</h1>",
        "<section class='note'>",
        "<h2>Decision inicial de variables</h2>",
        "<p>Segun la guia, no conviene usar las 320 columnas como variables directas de clustering. "
        "El modelado debe partir de variables fisicas, orbitales, estelares y de habitabilidad; "
        "las demas columnas sirven para auditoria, calidad, sesgos, incertidumbre o trazabilidad.</p>",
        "<p>Este reporte por eso hace dos cosas: perfila todas las columnas y despues se concentra en las variables clave "
        "para interpretar distribuciones, correlaciones y cobertura antes de clustering.</p>",
        "</section>",
        derived_note,
        "<section>",
        "<h2>Resumen del dataset</h2>",
        "<div class='grid'>",
        f"<div class='metric'><span>Archivo</span><strong>{html.escape(csv_path.name)}</strong></div>",
        f"<div class='metric'><span>Filas</span><strong>{df.shape[0]:,}</strong></div>",
        f"<div class='metric'><span>Columnas</span><strong>{df.shape[1]:,}</strong></div>",
        f"<div class='metric'><span>Numericas</span><strong>{df.select_dtypes(include='number').shape[1]:,}</strong></div>",
        "</div>",
        f"<p>Columnas con al menos 50% nulos: <code>{missing_over_50}</code>. "
        f"Columnas con al menos 80% nulos: <code>{missing_over_80}</code>.</p>",
        "</section>",
        "<section><h2>Nulos en variables clave</h2>",
        "<div class='table-wrap'>",
        dataframe_html(key_missing, max_rows=40),
        "</div></section>",
        "<section><h2>Rangos y cuantiles de variables clave</h2>",
        "<div class='table-wrap'>",
        dataframe_html(key_stats, max_rows=40),
        "</div></section>",
        "<section><h2>Cobertura para clustering</h2>",
        "<div class='table-wrap'>",
        dataframe_html(coverage, max_rows=20),
        "</div></section>",
        "<section><h2>Correlaciones fuertes</h2>",
        "<div class='table-wrap'>",
        dataframe_html(strong_corr, max_rows=40),
        "</div></section>",
    ]

    for snippet in snippets:
        html_parts.append(f"<section>{snippet}</section>")

    html_parts.extend(
        [
            "<section><h2>Perfil numerico: mayores nulos</h2>",
            "<div class='table-wrap'>",
            dataframe_html(numeric_all, max_rows=60),
            "</div></section>",
            "<section><h2>Resumen JSON</h2>",
            f"<pre>{html.escape(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>",
            "</section>",
            "</main></body></html>",
        ]
    )
    return "\n".join(html_parts)


def write_outputs(csv_path: Path, df: pd.DataFrame, reports_dir: Path) -> dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    descriptions = parse_column_descriptions(csv_path)
    if "pl_dens" in df.attrs.get("derived_columns", {}):
        descriptions["pl_dens"] = "Planet Density [g/cm3], derived from pl_bmasse and pl_rade"
        descriptions["pl_dens_source"] = "Source marker for derived planet density"
    profile = column_profile(df, descriptions)
    numeric_all = numeric_profile(df, descriptions)
    key_stats = key_variable_stats(df, descriptions)
    coverage = clustering_coverage(df)

    core_cols = [col for col in CONTEST_NUMERIC_COLUMNS if col in df.columns]
    corr_core = df[core_cols].corr(method="spearman", min_periods=30)

    candidate_cols = candidate_numeric_columns(df)
    corr_candidates = df[candidate_cols].corr(method="spearman", min_periods=30)
    strong_corr = strong_correlations(corr_candidates)

    data_dictionary = profile[["column", "description", "dtype", "missing_pct", "unique"]].copy()
    data_dictionary["contest_key"] = data_dictionary["column"].isin(CONTEST_KEY_COLUMNS)
    data_dictionary["model_auxiliary"] = data_dictionary["column"].map(is_model_auxiliary)
    duplicate_lookup = {
        col: group for group, columns in DUPLICATE_UNIT_GROUPS.items() for col in columns
    }
    data_dictionary["duplicate_unit_group"] = data_dictionary["column"].map(duplicate_lookup).fillna("")

    paths = {
        "profile": reports_dir / "missingness_all_columns.csv",
        "numeric_profile": reports_dir / "numeric_profile_all_columns.csv",
        "key_stats": reports_dir / "key_variable_stats.csv",
        "coverage": reports_dir / "clustering_feature_coverage.csv",
        "corr_core": reports_dir / "correlation_spearman_core.csv",
        "corr_candidates": reports_dir / "correlation_spearman_candidates.csv",
        "strong_corr": reports_dir / "strong_correlations.csv",
        "data_dictionary": reports_dir / "data_dictionary.csv",
        "summary_json": reports_dir / "eda_summary.json",
        "html": reports_dir / "exodata_eda_plotly.html",
    }

    profile.to_csv(paths["profile"], index=False)
    numeric_all.to_csv(paths["numeric_profile"], index=False)
    key_stats.to_csv(paths["key_stats"], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    corr_core.to_csv(paths["corr_core"])
    corr_candidates.to_csv(paths["corr_candidates"])
    strong_corr.to_csv(paths["strong_corr"], index=False)
    data_dictionary.to_csv(paths["data_dictionary"], index=False)

    summary = {
        "csv_path": str(csv_path),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "numeric_columns": int(df.select_dtypes(include="number").shape[1]),
        "derived_columns": df.attrs.get("derived_columns", {}),
        "contest_key_columns_available": [col for col in CONTEST_KEY_COLUMNS if col in df.columns],
        "clustering_feature_sets": coverage.to_dict(orient="records"),
        "recommendation": (
            "Usa todas las columnas para auditoria, pero empieza clustering con variables del PDF. "
            "Evita IDs, enlaces, errores/limites y duplicados de unidad como pl_radj/pl_bmassj."
        ),
    }
    paths["summary_json"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report_html = build_report(
        csv_path=csv_path,
        df=df,
        profile=profile,
        key_stats=key_stats,
        numeric_all=numeric_all,
        coverage=coverage,
        corr_core=corr_core,
        corr_candidates=corr_candidates,
        strong_corr=strong_corr,
        descriptions=descriptions,
    )
    paths["html"].write_text(report_html, encoding="utf-8")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera EDA Plotly para ExoData Challenge.")
    parser.add_argument("--csv", default=None, help="Ruta al CSV PSCompPars. Si se omite, se autodetecta.")
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="Carpeta de salida. Si se omite, usa reports/<nombre-del-csv>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = find_csv(args.csv)
    reports_dir = Path(args.reports_dir) if args.reports_dir else REPORTS_DIR / csv_path.stem
    if not reports_dir.is_absolute():
        reports_dir = PROJECT_ROOT / reports_dir

    df = load_data(csv_path)
    paths = write_outputs(csv_path, df, reports_dir)

    print("EDA generado correctamente.")
    print(f"CSV: {csv_path}")
    print(f"Filas x columnas: {df.shape[0]} x {df.shape[1]}")
    print(f"Reporte HTML: {paths['html']}")
    print("Tablas:")
    for key, path in paths.items():
        if key != "html":
            print(f"  - {path}")


if __name__ == "__main__":
    main()
