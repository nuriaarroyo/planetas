from __future__ import annotations

import argparse

from mapper_tda.io import find_mapper_input_csv, load_mapper_input, resolve_reports_dir, write_mapper_outputs
from mapper_tda.pipeline import expand_configs_from_cli, run_mapper_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye y compara grafos Mapper/TDA para espacios fisicos, orbitales y conjuntos.",
    )
    parser.add_argument("--csv", default=None, help="Ruta opcional al CSV de entrada.")
    parser.add_argument(
        "--reports-dir",
        default="reports/mapper",
        help="Carpeta de salida. Default: reports/mapper.",
    )
    parser.add_argument(
        "--space",
        default="all",
        choices=["phys", "orb", "joint", "all"],
        help="Espacio de variables a analizar.",
    )
    parser.add_argument(
        "--lens",
        default="pca2",
        choices=["pca2", "density", "domain", "all"],
        help="Lens a construir.",
    )
    parser.add_argument("--n-cubes", type=int, default=10, help="Numero de cubos del cover.")
    parser.add_argument("--overlap", type=float, default=0.35, help="Overlap del cover.")
    parser.add_argument(
        "--clusterer",
        default="dbscan",
        choices=["dbscan"],
        help="Clusterer local. Por ahora solo dbscan.",
    )
    parser.add_argument("--min-samples", type=int, default=4, help="Parametro min_samples para DBSCAN.")
    parser.add_argument(
        "--eps-percentile",
        type=float,
        default=90.0,
        help="Percentil para estimar eps de DBSCAN a partir de k-distances.",
    )
    parser.add_argument("--k-density", type=int, default=15, help="Vecino k para el lens de densidad local.")
    parser.add_argument("--grid", action="store_true", help="Correr la grilla de covers n_cubes x overlap.")
    parser.add_argument(
        "--complete-case-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Eliminar filas con NaN en el espacio seleccionado. Default: true.",
    )
    parser.add_argument(
        "--include-eccentricity",
        action="store_true",
        help="Agregar pl_orbeccen al espacio orbital y conjunto.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Semilla reproducible.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = find_mapper_input_csv(args.csv)
    reports_dir = resolve_reports_dir(args.reports_dir)
    configs = expand_configs_from_cli(args)
    df = load_mapper_input(csv_path)
    batch_result = run_mapper_batch(df, configs, csv_path=csv_path, grid_mode=args.grid)
    paths = write_mapper_outputs(batch_result, reports_dir)

    print("Mapper/TDA generado correctamente.")
    print(f"CSV: {csv_path}")
    print(f"Filas de entrada: {len(df):,}")
    print(f"Configuraciones corridas: {len(configs)}")
    print("Salidas principales:")
    for key in [
        "mapper_graph_metrics",
        "mapper_graph_distances",
        "mapper_stability_grid",
        "mapper_config_summary",
        "mapper_report",
    ]:
        if key in paths:
            print(f"  - {paths[key]}")


if __name__ == "__main__":
    main()
