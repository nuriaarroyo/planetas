from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mapper_tda.region_synthesis import synthesize_regions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAPPER_DIR = PROJECT_ROOT / "outputs" / "mapper"
TABLES_DIR = MAPPER_DIR / "tables"
LATEX_DIR = PROJECT_ROOT / "latex" / "03_mapper"
LATEX_TABLES_DIR = LATEX_DIR / "tables"
LATEX_SECTIONS_DIR = LATEX_DIR / "sections"
LATEX_FIGURES_DIR = LATEX_DIR / "figures"
SUMMARY_PATH = TABLES_DIR / "latex_report_build_summary.md"


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required input: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def tex_escape(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def code(value: Any) -> str:
    return r"\texttt{" + tex_escape(value) + "}"


def fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return tex_escape(value)
    if not np.isfinite(number):
        return "NA"
    if abs(number - round(number)) < 1e-10 and abs(number) < 1_000_000:
        return str(int(round(number)))
    return f"{number:.{digits}f}"


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def truncate_text(value: Any, max_len: int = 95) -> str:
    text = "" if pd.isna(value) else str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def select_columns(frame: pd.DataFrame, columns: list[str], notes: list[str], table_name: str) -> pd.DataFrame:
    present = [column for column in columns if column in frame.columns]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        notes.append(f"{table_name}: omitted missing columns: {', '.join(missing)}.")
    return frame.loc[:, present].copy() if present else pd.DataFrame()


def write_latex_table(
    frame: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
    notes: list[str] | None = None,
    max_rows: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    notes = notes or []
    table = frame.copy()
    if max_rows is not None and len(table) > max_rows:
        notes.append(f"Se muestran {max_rows} de {len(table)} filas; el CSV fuente contiene la tabla completa.")
        table = table.head(max_rows).copy()
    if table.empty:
        body = "% No measured rows available for this table.\n"
    else:
        for column in table.columns:
            if table[column].dtype == object:
                table[column] = table[column].map(lambda value: truncate_text(value))
        body = table.to_latex(index=False, escape=True, float_format=lambda value: f"{value:.3f}", na_rep="NA")
    note_text = ""
    if notes:
        note_text = "\n\\vspace{0.4em}\\begin{minipage}{0.96\\linewidth}\\footnotesize " + tex_escape(" ".join(notes)) + "\\end{minipage}\n"
    content = "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\scriptsize",
            f"\\caption{{{tex_escape(caption)}}}",
            f"\\label{{{label}}}",
            "\\begin{adjustbox}{max width=\\linewidth}",
            body.strip(),
            "\\end{adjustbox}",
            note_text.strip(),
            "\\end{table}",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def copy_existing_figures() -> list[str]:
    copied: list[str] = []
    LATEX_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    source_dirs = [MAPPER_DIR / "figures_pdf", MAPPER_DIR / "figures_pdf" / "interpretation"]
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for source in source_dir.glob("*.pdf"):
            target = LATEX_FIGURES_DIR / source.name
            if not target.exists() or source.stat().st_mtime > target.stat().st_mtime:
                shutil.copyfile(source, target)
                copied.append(source.name)
    return copied


def figure_tex(filename: str, caption: str, width: str = "0.92\\linewidth") -> str:
    path = LATEX_FIGURES_DIR / filename
    if not path.exists():
        return ""
    return "\n".join(
        [
            "\\begin{figure}[H]",
            "\\centering",
            f"\\includegraphics[width={width}]{{figures/{filename}}}",
            f"\\caption{{{tex_escape(caption)}}}",
            "\\end{figure}",
            "",
        ]
    )


def unique_cover_setting(manifest: pd.DataFrame) -> str:
    settings = manifest[["n_cubes", "overlap"]].drop_duplicates() if {"n_cubes", "overlap"}.issubset(manifest.columns) else pd.DataFrame()
    if len(settings) == 1:
        row = settings.iloc[0]
        return f"cubes{int(float(row['n_cubes']))}_overlap{str(row['overlap']).replace('.', 'p')}"
    return ", ".join(f"cubes{row.n_cubes}_overlap{row.overlap}" for row in settings.itertuples())


def ensure_final_region_synthesis() -> pd.DataFrame:
    path = TABLES_DIR / "final_region_synthesis.csv"
    if path.exists():
        return read_csv(path)
    synthesis, _, _ = synthesize_regions(MAPPER_DIR)
    return synthesis


def check_permutation_final(permutation: pd.DataFrame) -> int:
    if "n_perm" not in permutation.columns or permutation.empty:
        raise SystemExit("discoverymethod_permutation_null.csv does not contain n_perm. Rerun the bias audit.")
    max_perm = int(pd.to_numeric(permutation["n_perm"], errors="coerce").max())
    if max_perm < 1000:
        raise SystemExit(
            "The available discoverymethod_permutation_null.csv is a smoke run "
            f"(n_perm={max_perm}). Rerun:\npython .\\src\\evaluate_mapper_bias.py --n-perm 1000 --seed 42"
        )
    return max_perm


def table_output_manifest_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    complete = bool_series(manifest["has_graph"]) & bool_series(manifest["has_nodes"]) & bool_series(manifest["has_edges"]) & bool_series(manifest["has_config"])
    rows = [
        ("graph_json_files", int(bool_series(manifest["has_graph"]).sum())),
        ("node_csv_files", int(bool_series(manifest["has_nodes"]).sum())),
        ("edge_csv_files", int(bool_series(manifest["has_edges"]).sum())),
        ("config_json_files", int(bool_series(manifest["has_config"]).sum())),
        ("complete_graph_node_edge_config_sets", int(complete.sum())),
        ("feature_spaces", int(manifest["feature_space"].nunique())),
        ("lenses", int(manifest["lens"].nunique())),
        ("cover_setting", unique_cover_setting(manifest)),
    ]
    return pd.DataFrame(rows, columns=["metric", "measured_value"])


def build_tables(inputs: dict[str, pd.DataFrame]) -> dict[str, Path]:
    LATEX_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    notes: list[str] = []
    path = LATEX_TABLES_DIR / "table_output_manifest_summary.tex"
    write_latex_table(table_output_manifest_summary(inputs["manifest"]), path, "Resumen medido de artefactos Mapper existentes.", "tab:output_manifest_summary")
    outputs[path.name] = path

    main = inputs["main_selection"].merge(
        inputs["metrics_all"][["config_id", "beta_0"]] if "beta_0" in inputs["metrics_all"].columns else pd.DataFrame(columns=["config_id", "beta_0"]),
        on="config_id",
        how="left",
    )
    main = main.rename(columns={"interpretation_priority": "priority", "caution_level": "caution", "mean_node_imputation_fraction": "mean_node_imputation"})
    cols = ["config_id", "priority", "caution", "n_nodes", "n_edges", "beta_0", "beta_1", "mean_node_imputation"]
    path = LATEX_TABLES_DIR / "table_main_graph_selection.tex"
    write_latex_table(select_columns(main, cols, notes, path.name), path, "Grafos seleccionados para interpretacion.", "tab:main_graph_selection", notes.copy())
    outputs[path.name] = path

    notes = []
    metrics = inputs["metrics_all"].rename(columns={"mean_node_imputation_fraction": "mean_node_imputation"})
    cols = ["config_id", "feature_space", "lens", "n_nodes", "n_edges", "beta_0", "beta_1", "mean_node_size", "mean_node_imputation"]
    path = LATEX_TABLES_DIR / "table_all_existing_mapper_metrics.tex"
    write_latex_table(select_columns(metrics, cols, notes, path.name), path, "Metricas reconstruidas para todos los artefactos Mapper existentes.", "tab:all_existing_mapper_metrics", notes.copy())
    outputs[path.name] = path

    notes = []
    lens = inputs["lens_all"].rename(columns={"mean_node_imputation_fraction": "mean_node_imputation"})
    cols = ["feature_space", "lens", "n_nodes", "n_edges", "beta_1", "mean_node_imputation"]
    path = LATEX_TABLES_DIR / "table_lens_sensitivity.tex"
    write_latex_table(select_columns(lens, cols, notes, path.name), path, "Sensibilidad por lens usando todos los artefactos existentes.", "tab:lens_sensitivity", notes.copy())
    outputs[path.name] = path

    notes = []
    space = inputs["space_all"].rename(columns={"mean_node_imputation_fraction": "mean_node_imputation"})
    cols = ["feature_space", "lens", "beta_1", "n_nodes", "n_edges", "mean_node_imputation"]
    path = LATEX_TABLES_DIR / "table_space_comparison.tex"
    write_latex_table(select_columns(space, cols, notes, path.name), path, "Comparacion de espacios de variables para los artefactos existentes.", "tab:space_comparison", notes.copy())
    outputs[path.name] = path

    notes = []
    perm = inputs["permutation"]
    cols = [
        "config_id",
        "observed_weighted_mean_method_purity",
        "null_mean_purity",
        "purity_z",
        "purity_empirical_p",
        "observed_nmi",
        "null_mean_nmi",
        "nmi_z",
        "nmi_empirical_p",
    ]
    path = LATEX_TABLES_DIR / "table_bias_permutation_null.tex"
    write_latex_table(select_columns(perm, cols, notes, path.name), path, "Prueba nula por permutacion de etiquetas discoverymethod.", "tab:bias_permutation_null", notes.copy())
    outputs[path.name] = path

    notes = []
    enrich = inputs["enrichment"].merge(
        inputs["node_bias"][["config_id", "node_id", "n_members"]] if {"config_id", "node_id", "n_members"}.issubset(inputs["node_bias"].columns) else pd.DataFrame(),
        on=["config_id", "node_id"],
        how="left",
    )
    enrich = enrich.sort_values(["empirical_p_value", "enrichment_z"], ascending=[True, False], na_position="last")
    cols = ["config_id", "node_id", "dominant_discoverymethod", "observed_dominant_method_fraction", "enrichment_z", "empirical_p_value", "n_members"]
    path = LATEX_TABLES_DIR / "table_top_enriched_nodes.tex"
    write_latex_table(select_columns(enrich, cols, notes, path.name), path, "Nodos con mayor enriquecimiento por metodo de descubrimiento.", "tab:top_enriched_nodes", notes.copy(), max_rows=15)
    outputs[path.name] = path

    notes = []
    comp = inputs["component_bias"].sort_values(["discoverymethod_js_divergence_vs_global", "dominant_discoverymethod_fraction"], ascending=[False, False], na_position="last")
    cols = ["config_id", "component_id", "n_members", "dominant_discoverymethod", "dominant_discoverymethod_fraction", "discoverymethod_entropy", "dominant_disc_facility", "disc_year_median"]
    path = LATEX_TABLES_DIR / "table_component_discovery_bias.tex"
    write_latex_table(select_columns(comp, cols, notes, path.name), path, "Componentes con mayor concentracion observacional.", "tab:component_discovery_bias", notes.copy(), max_rows=15)
    outputs[path.name] = path

    notes = []
    final = inputs["final_regions"].copy()
    label_order = {"physical": 0, "observational": 1, "mixed": 2, "weak": 3}
    final["_label_order"] = final["final_label"].map(label_order).fillna(9)
    final = (
        final.sort_values(["_label_order", "confidence", "observational_bias_score", "physical_evidence_score"], ascending=[True, True, False, False])
        .groupby("final_label", group_keys=False)
        .head(8)
    )
    cols = [
        "config_id",
        "region_type",
        "region_id",
        "n_members",
        "final_label",
        "confidence",
        "dominant_discoverymethod_fraction",
        "discoverymethod_enrichment_z",
        "mean_imputation_fraction",
        "rationale_short",
    ]
    path = LATEX_TABLES_DIR / "table_final_region_synthesis.tex"
    write_latex_table(select_columns(final, cols, notes, path.name), path, "Sintesis final de regiones Mapper por tipo de evidencia.", "tab:final_region_synthesis", notes.copy(), max_rows=32)
    outputs[path.name] = path
    return outputs


def get_graph_used_features(config_id: str) -> list[str]:
    path = MAPPER_DIR / "graphs" / f"graph_{config_id}.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("mapper_metadata", {})
    features = metadata.get("used_features", [])
    return [str(feature) for feature in features] if isinstance(features, list) else []


def measured_context(inputs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    manifest = inputs["manifest"]
    main = inputs["main_selection"]
    metrics = inputs["metrics_all"]
    final_regions = inputs["final_regions"]
    permutation = inputs["permutation"]
    physical_derivations = inputs["physical_derivations"]
    coverage = inputs["coverage"]
    density = inputs["density_sensitivity"]

    complete = bool_series(manifest["has_graph"]) & bool_series(manifest["has_nodes"]) & bool_series(manifest["has_edges"]) & bool_series(manifest["has_config"])
    selected = main.set_index("config_id")
    metric_lookup = metrics.set_index("config_id")
    final_counts = final_regions.groupby(["region_type", "final_label"]).size().to_dict()
    orbital_id = "orbital_pca2_cubes10_overlap0p35"
    thermal_id = "thermal_pca2_cubes10_overlap0p35"
    orbital = selected.loc[orbital_id] if orbital_id in selected.index else pd.Series(dtype=object)
    thermal = selected.loc[thermal_id] if thermal_id in selected.index else pd.Series(dtype=object)
    n_perm = int(pd.to_numeric(permutation["n_perm"], errors="coerce").max())
    context = {
        "n_graphs": int(bool_series(manifest["has_graph"]).sum()),
        "n_nodes": int(bool_series(manifest["has_nodes"]).sum()),
        "n_edges": int(bool_series(manifest["has_edges"]).sum()),
        "n_configs": int(bool_series(manifest["has_config"]).sum()),
        "n_complete_sets": int(complete.sum()),
        "n_spaces": int(manifest["feature_space"].nunique()),
        "spaces": ", ".join(sorted(manifest["feature_space"].dropna().astype(str).unique())),
        "n_lenses": int(manifest["lens"].nunique()),
        "lenses": ", ".join(sorted(manifest["lens"].dropna().astype(str).unique())),
        "cover": unique_cover_setting(manifest),
        "selected_count": int(main["config_id"].nunique()),
        "n_perm": n_perm,
        "orbital_priority": str(orbital.get("interpretation_priority", "")),
        "orbital_caution": str(orbital.get("caution_level", "")),
        "orbital_beta1": fmt(orbital.get("beta_1")),
        "orbital_imputation": fmt(orbital.get("mean_node_imputation_fraction")),
        "orbital_features": ", ".join(get_graph_used_features(orbital_id)),
        "thermal_beta1": fmt(thermal.get("beta_1")),
        "thermal_imputation": fmt(thermal.get("mean_node_imputation_fraction")),
        "thermal_high_imputation": fmt(thermal.get("frac_nodes_high_imputation")),
        "node_physical": int(final_counts.get(("node", "physical"), 0)),
        "node_observational": int(final_counts.get(("node", "observational"), 0)),
        "node_mixed": int(final_counts.get(("node", "mixed"), 0)),
        "node_weak": int(final_counts.get(("node", "weak"), 0)),
        "component_observational": int(final_counts.get(("component", "observational"), 0)),
        "component_mixed": int(final_counts.get(("component", "mixed"), 0)),
        "component_weak": int(final_counts.get(("component", "weak"), 0)),
    }
    if not physical_derivations.empty:
        row = physical_derivations.iloc[0]
        context.update(
            {
                "density_derived_count": fmt(row.get("density_derived_count")),
                "density_missing_before": fmt(row.get("density_missing_before")),
                "density_missing_after": fmt(row.get("density_missing_after")),
                "kepler_derived_count": fmt(row.get("kepler_derived_count")),
                "kepler_missing_after": fmt(row.get("kepler_missing_after")),
            }
        )
    if not coverage.empty:
        joint = coverage[coverage["feature_group"].astype(str).eq("MAPPER_JOINT_FEATURES")]
        optional = coverage[coverage["feature_group"].astype(str).str.contains("OPTIONAL", na=False)]
        context["joint_before_complete_pct"] = fmt(joint.iloc[0].get("before_complete_pct")) if not joint.empty else "NA"
        context["joint_after_complete_pct"] = fmt(joint.iloc[0].get("after_complete_pct")) if not joint.empty else "NA"
        context["optional_excluded_columns"] = str(optional.iloc[0].get("excluded_columns", "")) if not optional.empty else ""
    if not density.empty:
        density_rows = []
        for row in density.to_dict(orient="records"):
            density_rows.append(
                f"{row.get('comparison')}: delta beta_1={fmt(row.get('delta_beta_1'))}, "
                f"delta nodos={fmt(row.get('delta_n_nodes'))}"
            )
        context["density_sensitivity_text"] = "; ".join(density_rows)
    else:
        context["density_sensitivity_text"] = "No disponible."
    if orbital_id in metric_lookup.index:
        context["orbital_metric_n_nodes"] = fmt(metric_lookup.loc[orbital_id].get("n_nodes"))
        context["orbital_metric_n_edges"] = fmt(metric_lookup.loc[orbital_id].get("n_edges"))
    return context


def write_sections(context: dict[str, Any]) -> dict[str, Path]:
    LATEX_SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    sections: dict[str, str] = {}

    sections["00_abstract.tex"] = rf"""
\begin{{abstract}}
Este reporte presenta un analisis Mapper/TDA exploratorio sobre datos de exoplanetas PSCompPars imputados. La evidencia se interpreta como resumen topologico de una matriz procesada, no como la topologia real de los exoplanetas ni como una taxonomia final. Los artefactos reconciliados contienen {context['n_graphs']} grafos, {context['n_nodes']} tablas de nodos, {context['n_edges']} tablas de aristas y {context['n_configs']} configuraciones, con {context['n_complete_sets']} conjuntos completos grafo--nodos--aristas--configuracion. Estos artefactos cubren {context['n_spaces']} espacios de variables y {context['n_lenses']} lenses con cubierta fija {code(context['cover'])}. La interpretacion final separa senal fisica/orbital, sesgo observacional, dependencia de imputacion y sensibilidad al lens.
\end{{abstract}}
"""

    sections["01_introduction.tex"] = rf"""
La pregunta de investigacion es si los grafos Mapper construidos sobre variables fisicas y orbitales de exoplanetas contienen regiones candidatas para inspeccion cientifica, y en que medida esas regiones pueden confundirse con sesgo observacional o dependencia de imputacion. La lectura es deliberadamente cautelosa: Mapper se usa como resumen exploratorio de vecindades inducidas por una matriz completada, no como prueba definitiva de estructura fisica.

La separacion central del reporte es metodologica. Una region puede ser compatible con una senal astrofisica, estar enriquecida por {code('discoverymethod')}, depender de variables imputadas, o cambiar al modificar el lens. Por eso las conclusiones se expresan como evidencia medida y no como clasificacion final de planetas.
"""

    sections["02_astrophysical_context.tex"] = rf"""
Los espacios de variables existentes son {code(context['spaces'])}. El espacio orbital seleccionado usa las variables medidas en el JSON del grafo {code('orbital_pca2_cubes10_overlap0p35')}: {code(context['orbital_features'])}. En particular, para estos artefactos reconciliados el espacio orbital no usa {code('pl_orbeccen')}. La excentricidad aparece como variable disponible u opcional en la etapa de datos, pero no como coordenada del Mapper orbital actual.

La densidad planetaria debe tratarse como control y no como observacion independiente cuando se usa junto con masa y radio. En la auditoria de derivaciones, {context.get('density_derived_count', 'NA')} valores de {code('pl_dens')} fueron derivados a partir de masa y radio, con {context.get('density_missing_before', 'NA')} faltantes antes de la derivacion y {context.get('density_missing_after', 'NA')} despues. Esta dependencia algebraica justifica comparar espacios con y sin densidad.
"""

    sections["03_data_and_imputation.tex"] = rf"""
La imputacion disponible se encuentra bajo {code('reports/imputation')}. La tabla de cobertura reporta que el grupo conjunto principal tenia {context.get('joint_before_complete_pct', 'NA')}\% de filas completas antes de imputar y {context.get('joint_after_complete_pct', 'NA')}\% despues. La variable semieje mayor tambien se derivo fisicamente cuando fue posible: {context.get('kepler_derived_count', 'NA')} valores de {code('pl_orbsmax')} fueron derivados y quedaron {context.get('kepler_missing_after', 'NA')} faltantes despues de esa derivacion.

El grupo opcional que incluye {code('pl_orbeccen')} no quedo como entrada de Mapper en los artefactos actuales; la tabla de cobertura marca como excluida la columna {code(context.get('optional_excluded_columns', ''))}. En consecuencia, las conclusiones orbitales de este reporte se limitan a periodo orbital y semieje mayor, no a dinamica orbital completa.
"""

    sections["04_mapper_methodology.tex"] = r"""
Mapper construye un grafo a partir de una nube de puntos, un lens y una cubierta del espacio del lens. En cada elemento de la cubierta se clusteriza la preimagen y se conectan nodos cuando comparten puntos. Para un grafo con conjunto de vertices \(V\), aristas \(E\) y numero de componentes conexas \(\beta_0\), el conteo de ciclos usado en este reporte es
\[
\beta_1 = |E| - |V| + \beta_0.
\]

Los lenses reconciliados son \texttt{pca2}, \texttt{domain} y \texttt{density}. No se encontro evidencia de una busqueda completa sobre \(n\_cubes\) y traslape; los artefactos existentes usan cubierta fija \texttt{cubes10\_overlap0p35}. Por tanto, las comparaciones de este reporte son comparaciones de espacios y lenses bajo una cubierta fija.
"""

    sections["05_output_reconciliation.tex"] = rf"""
La reconciliacion de salidas encontro {context['n_graphs']} JSON de grafos, {context['n_nodes']} CSV de nodos, {context['n_edges']} CSV de aristas y {context['n_configs']} JSON de configuracion. Todos los grafos tienen archivos correspondientes de nodos, aristas y configuracion. Estos artefactos corresponden a {context['n_spaces']} espacios de variables por {context['n_lenses']} lenses, todos con {code(context['cover'])}.

Las tablas agregadas originales activas estaban restringidas a {code('pca2')}. La reconciliacion reconstruyo metricas para todos los artefactos existentes y las escribio en {code('mapper_graph_metrics_all_existing.csv')}, {code('mapper_lens_sensitivity_all_existing.csv')} y {code('mapper_space_comparison_all_existing.csv')}. Esto evita confundir una corrida de interpretacion {code('pca2')} con una busqueda de grilla completa.

\input{{tables/table_output_manifest_summary.tex}}
"""

    sections["06_mapper_results.tex"] = rf"""
La seleccion principal contiene {context['selected_count']} grafos {code('pca2')}. El grafo {code('orbital_pca2_cubes10_overlap0p35')} aparece en la tabla de seleccion con prioridad {code(context['orbital_priority'])}, cautela {code(context['orbital_caution'])}, \(\beta_1={context['orbital_beta1']}\) y fraccion media de imputacion nodal {context['orbital_imputation']}. Esta evidencia lo convierte en candidato de inspeccion, no en prueba fisica definitiva.

El espacio termico debe interpretarse con cautela: {code('thermal_pca2_cubes10_overlap0p35')} tiene \(\beta_1={context['thermal_beta1']}\), fraccion media de imputacion nodal {context['thermal_imputation']} y fraccion de nodos de alta imputacion {context['thermal_high_imputation']}. La complejidad termica observada es compatible con estructura inducida por variables completadas y no debe sobreinterpretarse.

La sensibilidad a densidad medida fue: {tex_escape(context.get('density_sensitivity_text', 'No disponible.'))}. Esto es coherente con tratar {code('pl_dens')} como variable de control derivada.

{figure_tex('01_mapper_graph_size_complexity.pdf', 'Tamano y complejidad de los grafos Mapper reconciliados.')}
{figure_tex('02_mapper_metrics_zscore_heatmap.pdf', 'Metricas Mapper estandarizadas para comparacion exploratoria.')}
{figure_tex('04_mapper_nodes_vs_cycles.pdf', 'Relacion entre numero de nodos y ciclos.')}
\input{{tables/table_main_graph_selection.tex}}
"""

    sections["07_selected_graphs.tex"] = rf"""
Los grafos seleccionados se interpretan como regiones candidatas, no como clases finales. La seleccion incluye controles fisicos con y sin densidad, el espacio orbital, el espacio conjunto con y sin densidad y el espacio termico de cautela. La comparacion completa de artefactos existentes muestra que la interpretacion principal sigue siendo {code('pca2')}, mientras que {code('domain')} y {code('density')} sirven como sensibilidad.

{figure_tex('04_orbital_mapper_interpretation.pdf', 'Interpretacion del Mapper orbital seleccionado.')}
{figure_tex('05_joint_mapper_interpretation.pdf', 'Interpretacion del Mapper conjunto seleccionado.')}
{figure_tex('09_thermal_caution.pdf', 'Complejidad termica con cautela por imputacion.')}
\input{{tables/table_all_existing_mapper_metrics.tex}}
"""

    sections["08_astrophysical_interpretation.tex"] = rf"""
La evidencia astrofisica mas util aparece cuando una region combina baja imputacion, coherencia fisica u orbital y baja concentracion observacional. En los resultados finales hay {context['node_physical']} nodos etiquetados como {code('physical')}. En contraste, muchas regiones quedan como {code('mixed')}: {context['node_mixed']} nodos y {context['component_mixed']} componentes. Esto significa que tienen algun patron fisico u orbital, pero tambien senal observacional relevante.

Las regiones fisicamente interpretables deben leerse como candidatas para inspeccion, por ejemplo regiones con pureza de clase fisica alta y baja imputacion. La evidencia no permite hablar de una taxonomia final de exoplanetas.

{figure_tex('01_main_graphs_by_population.pdf', 'Grafos principales coloreados por poblacion fisica heuristica.')}
{figure_tex('02_main_graphs_by_imputation.pdf', 'Grafos principales coloreados por fraccion de imputacion.')}
\input{{tables/table_space_comparison.tex}}
"""

    sections["09_observational_bias_audit.tex"] = r"""
Las variables \texttt{discoverymethod}, \texttt{disc\_year} y \texttt{disc\_facility} se usan como metadata externa para auditar sesgo observacional. No son variables de entrada de Mapper. Para un nodo \(v\) con miembros \(S_v\), la distribucion por metodo de descubrimiento es
\[
p_v(m)=\frac{|\{i\in S_v : d_i=m\}|}{|S_v|}.
\]
La pureza por metodo es
\[
P_v = \max_m p_v(m),
\]
y la entropia por metodo es
\[
H_v=-\sum_m p_v(m)\log p_v(m).
\]

Una region con alta pureza, baja entropia, concentracion en una facilidad de descubrimiento o ventana temporal estrecha puede estar reflejando sesgo de observacion. Esta asociacion no prueba causalidad, pero obliga a separar estructura astrofisica de estructura inducida por el proceso de descubrimiento.

\input{tables/table_top_enriched_nodes.tex}
\input{tables/table_component_discovery_bias.tex}
"""

    sections["10_permutation_null_test.tex"] = rf"""
La prueba nula mantiene fija la topologia del grafo y las membresias de nodos, y permuta etiquetas {code('discoverymethod')} entre planetas. Para cada nodo se compara la pureza observada con la distribucion nula. El estadistico reportado es
\[
Z_v = \frac{{P_v^{{obs}}-\mathbb{{E}}[P_v^{{null}}]}}{{\operatorname{{sd}}(P_v^{{null}})+\epsilon}}.
\]

Tambien se reporta NMI, descrita aqui como informacion mutua normalizada entre una asignacion dura determinista de planeta a nodo y {code('discoverymethod')}. La asignacion dura usa el nodo mas grande al que pertenece cada planeta, y desempata por orden alfabetico del identificador del nodo. La auditoria final uso {context['n_perm']} permutaciones.

\input{{tables/table_bias_permutation_null.tex}}
"""

    sections["11_region_synthesis.tex"] = rf"""
La sintesis final clasifica cada nodo y componente seleccionado como {code('physical')}, {code('observational')}, {code('mixed')} o {code('weak')}. La regla combina cuatro fuentes: coherencia fisica/orbital/termica, enriquecimiento por metodo de descubrimiento, riesgo de imputacion y tamano de region.

Conteos finales en nodos: {context['node_physical']} {code('physical')}, {context['node_observational']} {code('observational')}, {context['node_mixed']} {code('mixed')} y {context['node_weak']} {code('weak')}. En componentes: {context['component_observational']} {code('observational')}, {context['component_mixed']} {code('mixed')} y {context['component_weak']} {code('weak')}. La abundancia de regiones {code('mixed')} indica que no se puede descartar sesgo observacional aun cuando exista coherencia fisica.

\input{{tables/table_final_region_synthesis.tex}}
"""

    sections["12_limitations.tex"] = r"""
Este analisis tiene limitaciones importantes. Primero, Mapper es exploratorio: resume una matriz procesada e imputada, pero no entrega la topologia real de los exoplanetas. Segundo, la cubierta reconciliada esta fija en 10 cubos y 0.35 de traslape; no se debe presentar como busqueda completa de hiperparametros. Tercero, \texttt{pl\_dens} puede estar derivada de masa y radio, por lo que no es evidencia independiente cuando se analiza junto con esas variables.

La prueba de permutacion de \texttt{discoverymethod} mide asociacion entre regiones Mapper y metadata observacional. No prueba causalidad. Una region enriquecida por metodo de descubrimiento puede corresponder a un sesgo instrumental, a una poblacion fisica que se descubre preferentemente por cierto metodo, o a una mezcla de ambas. Por eso el reporte usa categorias de evidencia y evita afirmar una clasificacion final de planetas.
"""

    sections["13_conclusion.tex"] = r"""
La evidencia sugiere que los grafos Mapper seleccionados son utiles para priorizar regiones de inspeccion, siempre que se separen cuatro capas: senal fisica u orbital, sesgo observacional, dependencia de imputacion y sensibilidad al lens. El grafo orbital \texttt{orbital\_pca2\_cubes10\_overlap0p35} es un candidato importante por su complejidad y baja imputacion, pero la auditoria de sesgo muestra asociacion fuerte con \texttt{discoverymethod}; por tanto, debe interpretarse como evidencia mixta, no como estructura fisica pura.

La complejidad termica es especialmente cautelosa por su alta imputacion. La densidad funciona como control metodologico porque muchos valores de \texttt{pl\_dens} son derivados de masa y radio. La conclusion principal no es una taxonomia final, sino una clasificacion reproducible de regiones: algunas fisicamente interpretables, algunas observacionalmente sospechosas, muchas mixtas y otras debiles. Este resultado establece una base medible para una siguiente etapa de validacion, sin sobreafirmar la topologia real del catalogo.
"""

    outputs = {}
    for name, content in sections.items():
        path = LATEX_SECTIONS_DIR / name
        path.write_text(content.strip() + "\n", encoding="utf-8")
        outputs[name] = path
    return outputs


def write_main_tex() -> Path:
    content = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
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
\usepackage{url}

\title{Reporte Mapper/TDA de exoplanetas PSCompPars imputados}
\author{}
\date{}

\begin{document}
\maketitle
\input{sections/00_abstract.tex}
\section{Introduccion y pregunta de investigacion}
\input{sections/01_introduction.tex}
\section{Contexto astrofisico y variables}
\input{sections/02_astrophysical_context.tex}
\section{Datos, imputacion y cantidades derivadas}
\input{sections/03_data_and_imputation.tex}
\section{Metodologia Mapper}
\input{sections/04_mapper_methodology.tex}
\section{Reconciliacion de salidas y reproducibilidad}
\input{sections/05_output_reconciliation.tex}
\section{Resultados principales de Mapper}
\input{sections/06_mapper_results.tex}
\section{Interpretacion de grafos seleccionados}
\input{sections/07_selected_graphs.tex}
\section{Interpretacion astrofisica}
\input{sections/08_astrophysical_interpretation.tex}
\section{Auditoria de sesgo observacional}
\input{sections/09_observational_bias_audit.tex}
\section{Prueba nula por permutacion de discoverymethod}
\input{sections/10_permutation_null_test.tex}
\section{Sintesis final de regiones}
\input{sections/11_region_synthesis.tex}
\section{Limitaciones}
\input{sections/12_limitations.tex}
\section{Conclusion}
\input{sections/13_conclusion.tex}
\FloatBarrier
\clearpage
\begin{landscape}
\section*{Tablas complementarias}
\input{tables/table_lens_sensitivity.tex}
\end{landscape}
\end{document}
"""
    path = LATEX_DIR / "mapper_report.tex"
    path.write_text(content, encoding="utf-8")
    return path


def load_inputs() -> dict[str, pd.DataFrame]:
    final_regions = ensure_final_region_synthesis()
    inputs = {
        "manifest": read_csv(TABLES_DIR / "output_manifest.csv"),
        "main_selection": read_csv(TABLES_DIR / "main_graph_selection.csv"),
        "metrics_all": read_csv(TABLES_DIR / "mapper_graph_metrics_all_existing.csv"),
        "lens_all": read_csv(TABLES_DIR / "mapper_lens_sensitivity_all_existing.csv"),
        "space_all": read_csv(TABLES_DIR / "mapper_space_comparison_all_existing.csv"),
        "permutation": read_csv(TABLES_DIR / "discoverymethod_permutation_null.csv"),
        "enrichment": read_csv(TABLES_DIR / "discoverymethod_enrichment_summary.csv"),
        "component_bias": read_csv(TABLES_DIR / "component_discovery_bias.csv"),
        "node_bias": read_csv(TABLES_DIR / "node_discovery_bias.csv"),
        "final_regions": final_regions,
        "physical_derivations": read_csv(PROJECT_ROOT / "reports" / "imputation" / "physical_derivations.csv", required=False),
        "coverage": read_csv(PROJECT_ROOT / "reports" / "imputation" / "outputs" / "tables" / "mapper_coverage_summary.csv", required=False),
        "method_comparison": read_csv(PROJECT_ROOT / "reports" / "imputation" / "method_comparison.csv", required=False),
        "density_sensitivity": read_csv(TABLES_DIR / "mapper_density_feature_sensitivity.csv", required=False),
    }
    check_permutation_final(inputs["permutation"])
    return inputs


def write_summary(table_outputs: dict[str, Path], section_outputs: dict[str, Path], copied_figures: list[str], main_tex: Path, inputs: dict[str, pd.DataFrame]) -> None:
    figure_lines = (
        [f"- `latex/03_mapper/figures/{name}`" for name in sorted(copied_figures)]
        if copied_figures
        else ["- No figure files needed copying; existing figure files were reused."]
    )
    lines = [
        "# LaTeX Report Build Summary",
        "",
        "Generated report assets for `latex/03_mapper/` from measured CSV, JSON, and markdown outputs.",
        "",
        "## Inputs Used",
        "",
        "- `outputs/mapper/tables/output_manifest.csv`",
        "- `outputs/mapper/tables/output_consistency_warnings.md`",
        "- `outputs/mapper/tables/mapper_graph_metrics_all_existing.csv`",
        "- `outputs/mapper/tables/mapper_lens_sensitivity_all_existing.csv`",
        "- `outputs/mapper/tables/mapper_space_comparison_all_existing.csv`",
        "- `outputs/mapper/tables/main_graph_selection.csv`",
        "- `outputs/mapper/tables/discoverymethod_permutation_null.csv`",
        "- `outputs/mapper/tables/discoverymethod_enrichment_summary.csv`",
        "- `outputs/mapper/tables/component_discovery_bias.csv`",
        "- `outputs/mapper/tables/final_region_synthesis.csv`",
        "- `reports/imputation/physical_derivations.csv`",
        "- `reports/imputation/outputs/tables/mapper_coverage_summary.csv`",
        "",
        f"Permutation n_perm detected: {int(pd.to_numeric(inputs['permutation']['n_perm'], errors='coerce').max())}",
        "",
        "## Tables Generated",
        "",
        *[f"- `latex/03_mapper/tables/{name}`" for name in sorted(table_outputs)],
        "",
        "## Sections Generated",
        "",
        *[f"- `latex/03_mapper/sections/{name}`" for name in sorted(section_outputs)],
        "",
        "## Figures Copied/Updated",
        "",
        *figure_lines,
        "",
        "## Main TeX",
        "",
        f"- `{main_tex.relative_to(PROJECT_ROOT).as_posix()}`",
        "",
        "## Notes",
        "",
        "- Mapper was not rerun.",
        "- Feature-space definitions were not modified.",
        "- Tables omit optional columns only when the source CSV does not contain them.",
        "- The report frames Mapper as exploratory evidence, not as a final exoplanet taxonomy.",
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    copied_figures = copy_existing_figures()
    inputs = load_inputs()
    table_outputs = build_tables(inputs)
    context = measured_context(inputs)
    section_outputs = write_sections(context)
    main_tex = write_main_tex()
    write_summary(table_outputs, section_outputs, copied_figures, main_tex, inputs)
    print("Mapper LaTeX report assets generated.")
    print(f"Main TeX: {main_tex.relative_to(PROJECT_ROOT)}")
    print(f"Build summary: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
