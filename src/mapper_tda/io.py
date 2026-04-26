from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .pipeline import MapperConfig, build_stability_grid, config_id
from .report import build_mapper_report_html


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"


def find_mapper_input_csv(csv_arg: str | None) -> Path:
    if csv_arg:
        path = Path(csv_arg)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"No existe el CSV indicado: {path}")
        return path

    candidates = [
        REPORTS_DIR / "imputation" / "mapper_features_imputed_knn.csv",
        REPORTS_DIR / "imputation" / "mapper_features_complete_case.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    data_candidates = sorted(DATA_DIR.glob("PSCompPars_*.csv"))
    if not data_candidates:
        raise FileNotFoundError(
            "No encontré un CSV para Mapper. Probé reports/imputation/ y data/PSCompPars_*.csv."
        )
    return data_candidates[-1]


def resolve_reports_dir(value: str) -> Path:
    path = Path(value) if value else REPORTS_DIR / "mapper"
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_mapper_input(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, comment="#", low_memory=False)


def _float_slug(value: float) -> str:
    return str(value).replace(".", "p")


def _artifact_stem(config: MapperConfig) -> str:
    return f"{config.space}_{config.lens}_cubes{config.n_cubes}_overlap{_float_slug(config.overlap)}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _load_imputation_context() -> dict[str, Any]:
    imputation_dir = REPORTS_DIR / "imputation"
    context: dict[str, Any] = {
        "method_comparison": _read_optional_csv(imputation_dir / "method_comparison.csv"),
        "coverage_knn": _read_optional_csv(imputation_dir / "feature_coverage_before_after_knn.csv"),
        "missingness_before": _read_optional_csv(imputation_dir / "missingness_profile_before.csv"),
    }

    validation_frames: list[pd.DataFrame] = []
    for path in sorted(imputation_dir.glob("validation_metrics_*.csv")):
        frame = pd.read_csv(path)
        if "method" not in frame.columns:
            frame["method"] = path.stem.replace("validation_metrics_", "")
        validation_frames.append(frame)
    context["validation_metrics"] = pd.concat(validation_frames, ignore_index=True) if validation_frames else pd.DataFrame()

    return context


def write_mapper_outputs(result: dict, reports_dir: Path) -> dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    artifact_rows: list[dict[str, Any]] = []
    result["imputation_context"] = _load_imputation_context()

    for graph_result in result["results"]:
        config: MapperConfig = graph_result["config"]
        stem = _artifact_stem(config)

        graph_json_path = reports_dir / f"mapper_graph_{stem}.json"
        node_csv_path = reports_dir / f"mapper_nodes_{stem}.csv"
        edge_csv_path = reports_dir / f"mapper_edges_{stem}.csv"
        metrics_json_path = reports_dir / f"mapper_metrics_{stem}.json"
        html_path = reports_dir / f"mapper_{stem}.html"

        graph_payload = {
            "config": {**config.__dict__, "config_id": config_id(config)},
            "mapper_metadata": graph_result["mapper_metadata"],
            "graph_metrics": graph_result["graph_metrics"],
            "used_features": graph_result["used_features"],
            "lens_metadata": graph_result["lens_metadata"],
            "graph": {
                key: value
                for key, value in graph_result["graph"].items()
                if key in {"nodes", "links", "simplices", "meta_data", "meta_nodes", "sample_id_lookup"}
            },
        }
        write_json(graph_json_path, graph_payload)
        graph_result["node_table"].to_csv(node_csv_path, index=False)
        graph_result["edge_table"].to_csv(edge_csv_path, index=False)
        write_json(metrics_json_path, graph_result["graph_metrics"])
        html_path.write_text(graph_result["html"], encoding="utf-8")

        paths[f"mapper_graph_{stem}"] = graph_json_path
        paths[f"mapper_nodes_{stem}"] = node_csv_path
        paths[f"mapper_edges_{stem}"] = edge_csv_path
        paths[f"mapper_metrics_{stem}"] = metrics_json_path
        paths[f"mapper_html_{stem}"] = html_path

        artifact_rows.append(
            {
                "config_id": graph_result["config_id"],
                "space": config.space,
                "lens": config.lens,
                "n_cubes": config.n_cubes,
                "overlap": config.overlap,
                "html_path": html_path,
                "graph_json_path": graph_json_path,
                "node_csv_path": node_csv_path,
                "edge_csv_path": edge_csv_path,
                "metrics_json_path": metrics_json_path,
            }
        )

    metrics_csv_path = reports_dir / "mapper_graph_metrics.csv"
    distances_csv_path = reports_dir / "mapper_graph_distances.csv"
    config_summary_path = reports_dir / "mapper_config_summary.json"
    artifact_index_path = reports_dir / "mapper_artifacts.csv"
    report_path = reports_dir / "mapper_report.html"

    result["metrics_df"].to_csv(metrics_csv_path, index=False)
    result["distances_df"].to_csv(distances_csv_path, index=False)
    write_json(config_summary_path, result["config_summary"])

    paths["mapper_graph_metrics"] = metrics_csv_path
    paths["mapper_graph_distances"] = distances_csv_path
    paths["mapper_config_summary"] = config_summary_path

    if result.get("stability_grid_df") is not None and not result["stability_grid_df"].empty:
        stability_path = reports_dir / "mapper_stability_grid.csv"
        result["stability_grid_df"].to_csv(stability_path, index=False)
        paths["mapper_stability_grid"] = stability_path

    artifact_index = pd.DataFrame(artifact_rows)
    artifact_index.to_csv(artifact_index_path, index=False)
    paths["mapper_artifacts"] = artifact_index_path
    report_html = build_mapper_report_html(result, artifact_index)
    report_path.write_text(report_html, encoding="utf-8")
    paths["mapper_report"] = report_path
    return paths
