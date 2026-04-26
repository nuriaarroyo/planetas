from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from imputation.pipeline import ImputationConfig, run_imputation_pipeline
from imputation.steps.physical_derivation import (
    EARTH_DENSITY_G_CM3,
    apply_physical_derivations,
    derive_planet_density,
    derive_semimajor_axis,
)


def synthetic_pscomppars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pl_name": [f"p{i}" for i in range(8)],
            "hostname": [f"h{i}" for i in range(8)],
            "hd_name": [pd.NA] * 8,
            "discoverymethod": ["Transit", "RV", "Transit", "RV", "Transit", "RV", "Transit", "RV"],
            "disc_year": [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
            "disc_facility": ["A"] * 8,
            "tran_flag": [1, 0, 1, 0, 1, 0, 1, 0],
            "rv_flag": [0, 1, 0, 1, 0, 1, 0, 1],
            "ima_flag": [0] * 8,
            "micro_flag": [0] * 8,
            "sy_dist": [10.0, 20.0, 30.0, 40.0, np.nan, 60.0, 70.0, 80.0],
            "pl_rade": [1.0, 1.5, 2.0, 2.5, np.nan, 4.0, 5.0, 6.0],
            "pl_bmasse": [1.0, 5.0, 8.0, np.nan, 20.0, 40.0, 80.0, 120.0],
            "pl_dens": [np.nan, np.nan, np.nan, 2.5, np.nan, np.nan, np.nan, np.nan],
            "pl_orbper": [10.0, 20.0, np.nan, 50.0, 100.0, 200.0, 400.0, 800.0],
            "pl_orbsmax": [np.nan, np.nan, 0.2, np.nan, np.nan, np.nan, np.nan, np.nan],
            "pl_insol": [100.0, 50.0, 25.0, np.nan, 6.0, 3.0, 1.5, 0.7],
            "pl_eqt": [900.0, 700.0, 600.0, 500.0, np.nan, 350.0, 300.0, 250.0],
            "pl_orbeccen": [0.0, 0.1, np.nan, 0.2, 0.3, np.nan, 0.4, 0.5],
            "st_mass": [1.0, 1.1, 0.9, 1.0, 0.8, 1.2, 1.0, 0.7],
            "st_teff": [5800, 5700, 5600, 5500, 5400, np.nan, 5200, 5100],
            "st_met": [0.0, 0.1, -0.1, 0.2, np.nan, -0.2, 0.05, -0.05],
            "st_rad": [1.0, 1.1, 0.9, 1.0, 0.8, 1.2, 1.0, 0.7],
            "st_lum": [1.0, 1.2, 0.7, 1.1, 0.5, 1.5, 1.0, 0.3],
            "sy_pnum": [1, 1, 2, 2, 3, 3, 4, 4],
            "sy_snum": [1, 1, 1, 1, 2, 2, 1, 1],
            "pl_radeerr1": [0.1] * 8,
            "pl_bmass_reflink": ["ref"] * 8,
        }
    )


class PhysicalDerivationTests(unittest.TestCase):
    def test_pl_dens_is_derived_correctly(self) -> None:
        df = pd.DataFrame({"pl_rade": [2.0], "pl_bmasse": [8.0], "pl_dens": [np.nan]})
        out, audit = derive_planet_density(df)

        self.assertAlmostEqual(out.loc[0, "pl_dens"], EARTH_DENSITY_G_CM3)
        self.assertEqual(out.loc[0, "pl_dens_source"], "derived_density")
        self.assertEqual(audit.derived_count, 1)

    def test_pl_orbsmax_is_derived_with_kepler_approximation(self) -> None:
        df = pd.DataFrame({"pl_orbsmax": [np.nan], "pl_orbper": [365.25], "st_mass": [1.0]})
        out, audit = derive_semimajor_axis(df)

        self.assertAlmostEqual(out.loc[0, "pl_orbsmax"], 1.0)
        self.assertEqual(out.loc[0, "pl_orbsmax_source"], "derived_kepler")
        self.assertEqual(audit.derived_count, 1)


class PipelineTests(unittest.TestCase):
    def test_iterative_is_default_method_and_visualized_method(self) -> None:
        config = ImputationConfig()

        self.assertEqual(config.method, "iterative")
        self.assertEqual(config.visualized_method, "iterative")

    def test_indicators_exist_and_mapper_tables_have_no_nan_after_imputation(self) -> None:
        config = ImputationConfig(method="knn", n_neighbors=2, validation_mask_frac=0.2, random_state=11)
        result = run_imputation_pipeline(synthetic_pscomppars(), config)
        full = result.methods["knn"].full_df
        mapper = result.methods["knn"].mapper_tables["joint_imputed"]

        for feature in result.features_included:
            self.assertIn(f"{feature}_was_missing", full.columns)
            self.assertIn(f"{feature}_source", full.columns)
            self.assertIn(f"{feature}_was_observed", full.columns)
            self.assertIn(f"{feature}_was_physically_derived", full.columns)
            self.assertIn(f"{feature}_was_imputed", full.columns)
        self.assertFalse(mapper[result.features_included].isna().any().any())
        values = full[result.features_included].to_numpy()
        self.assertTrue(np.isfinite(values).all())
        for feature in ["pl_rade", "pl_bmasse", "pl_dens", "pl_orbper", "pl_orbsmax", "pl_insol"]:
            self.assertTrue((full[feature] > 0).all())

    def test_identifiers_and_references_do_not_enter_imputation_matrix(self) -> None:
        config = ImputationConfig(method="knn", n_neighbors=2)
        result = run_imputation_pipeline(synthetic_pscomppars(), config)

        forbidden = {"pl_name", "hostname", "hd_name", "pl_radeerr1", "pl_bmass_reflink", "discoverymethod"}
        self.assertTrue(forbidden.isdisjoint(result.features_included))

    def test_knn_is_reproducible_with_same_random_state(self) -> None:
        config = ImputationConfig(method="knn", n_neighbors=2, random_state=123)
        first = run_imputation_pipeline(synthetic_pscomppars(), config).methods["knn"].full_df
        second = run_imputation_pipeline(synthetic_pscomppars(), config).methods["knn"].full_df

        pd.testing.assert_frame_equal(first, second)

    def test_pipeline_does_not_mutate_input_dataframe(self) -> None:
        df = synthetic_pscomppars()
        original = df.copy(deep=True)
        run_imputation_pipeline(df, ImputationConfig(method="median"))

        pd.testing.assert_frame_equal(df, original)

    def test_script_runs_on_small_synthetic_csv_without_overwriting_original(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "PSCompPars_test.csv"
            reports_dir = tmp_path / "reports"
            synthetic_pscomppars().to_csv(csv_path, index=False)
            before = csv_path.read_text(encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "src" / "impute_exodata.py"),
                    "--csv",
                    str(csv_path),
                    "--reports-dir",
                    str(reports_dir),
                    "--method",
                    "compare",
                    "--visualized-method",
                    "iterative",
                    "--n-neighbors",
                    "2",
                    "--validation-mask-frac",
                    "0.2",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("Imputacion generada correctamente", completed.stdout)
            self.assertIn("Metodo visualizado: iterative", completed.stdout)
            self.assertEqual(csv_path.read_text(encoding="utf-8"), before)
            self.assertTrue((reports_dir / "PSCompPars_imputed_iterative.csv").exists())
            self.assertTrue((reports_dir / "mapper_features_imputed_iterative.csv").exists())
            report_html = (reports_dir / "imputation_report.html").read_text(encoding="utf-8")
            self.assertIn("METODO VISUALIZADO", report_html)
            self.assertIn("<strong>iterative</strong>", report_html)
            expected_pdfs = [
                "01_missingness_before_after.pdf",
                "02_value_source_composition.pdf",
                "03_mapper_coverage.pdf",
                "04_masked_validation_mae_by_feature.pdf",
                "05_masked_validation_spearman_heatmap.pdf",
                "06_method_comparison.pdf",
                "07_distribution_pl_rade.pdf",
                "08_distribution_pl_bmasse.pdf",
                "09_distribution_pl_dens.pdf",
                "10_distribution_pl_orbper.pdf",
                "11_distribution_pl_orbsmax.pdf",
                "12_distribution_pl_insol.pdf",
                "13_distribution_pl_eqt.pdf",
                "14_scatter_mass_radius.pdf",
                "15_scatter_density_radius.pdf",
                "16_scatter_orbper_orbsmax.pdf",
                "17_scatter_insol_eqt.pdf",
            ]
            for filename in expected_pdfs:
                path = reports_dir / "outputs" / "figures_pdf" / filename
                self.assertTrue(path.exists(), filename)
                self.assertGreater(path.stat().st_size, 0, filename)
            for filename in [
                "imputation_method_comparison.csv",
                "imputation_validation_metrics.csv",
                "imputation_value_source_composition.csv",
                "imputation_missingness_summary.csv",
                "mapper_coverage_summary.csv",
            ]:
                self.assertTrue((reports_dir / "outputs" / "tables" / filename).exists(), filename)


if __name__ == "__main__":
    unittest.main()
