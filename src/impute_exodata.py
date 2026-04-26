from __future__ import annotations

import argparse
from pathlib import Path

from imputation.io import PROJECT_ROOT, find_csv, load_pscomppars
from imputation.pipeline import ImputationConfig, run_imputation_pipeline, write_imputation_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Imputa valores faltantes de PSCompPars para Mapper/TDA y machine learning.",
    )
    parser.add_argument("--csv", default=None, help="Ruta al CSV. Si se omite, detecta el PSCompPars mas reciente.")
    parser.add_argument(
        "--reports-dir",
        default="reports/imputation",
        help="Carpeta de salida. Default: reports/imputation.",
    )
    parser.add_argument(
        "--method",
        default="knn",
        choices=["median", "knn", "iterative", "compare"],
        help="Metodo de imputacion o comparacion de metodos.",
    )
    parser.add_argument("--n-neighbors", type=int, default=15, help="Vecinos para KNNImputer.")
    parser.add_argument(
        "--weights",
        default="distance",
        choices=["uniform", "distance"],
        help="Pesos para KNNImputer.",
    )
    parser.add_argument(
        "--max-missing-pct",
        type=float,
        default=60.0,
        help="Excluir variables con mayor porcentaje de nulos en el espacio de modelado.",
    )
    parser.add_argument(
        "--validation-mask-frac",
        type=float,
        default=0.15,
        help="Fraccion de valores observados ocultados para validacion interna.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Semilla reproducible.")
    parser.add_argument(
        "--n-multiple-imputations",
        type=int,
        default=1,
        help="Numero de imputaciones multiples para method=iterative.",
    )
    parser.add_argument(
        "--include-stellar-context",
        action="store_true",
        help="Incluir st_teff, st_met, st_mass, st_rad, st_lum, sy_pnum y sy_snum.",
    )
    parser.add_argument(
        "--include-orbital-eccentricity",
        action="store_true",
        help="Incluir pl_orbeccen como variable dinamica opcional.",
    )
    return parser.parse_args()


def resolve_reports_dir(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    args = parse_args()
    csv_path = find_csv(args.csv)
    reports_dir = resolve_reports_dir(args.reports_dir)
    config = ImputationConfig(
        method=args.method,
        n_neighbors=args.n_neighbors,
        weights=args.weights,
        max_missing_pct=args.max_missing_pct,
        validation_mask_frac=args.validation_mask_frac,
        random_state=args.random_state,
        n_multiple_imputations=args.n_multiple_imputations,
        include_stellar_context=args.include_stellar_context,
        include_orbital_eccentricity=args.include_orbital_eccentricity,
    )

    df = load_pscomppars(csv_path)
    result = run_imputation_pipeline(df, config)
    paths = write_imputation_outputs(result, config, csv_path, reports_dir)

    print("Imputacion generada correctamente.")
    print(f"CSV: {csv_path}")
    print(f"Filas: {len(df):,}")
    print(f"Metodo solicitado: {args.method}")
    print(f"Features incluidas: {', '.join(result.features_included)}")
    if not result.excluded_features.empty:
        print("Features excluidas:")
        for row in result.excluded_features.to_dict(orient="records"):
            print(f"  - {row['feature']}: {row['reason']} ({row['missing_pct']}% missing)")
    print("Salidas principales:")
    for key in sorted(paths):
        if key.startswith("PSCompPars_imputed") or key.startswith("mapper_features") or key in {
            "method_comparison",
            "imputation_report",
        }:
            print(f"  - {paths[key]}")


if __name__ == "__main__":
    main()

