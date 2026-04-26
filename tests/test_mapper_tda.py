from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mapper_tda.cluster import estimate_dbscan_eps
from mapper_tda.lenses import make_lens_density, make_lens_pca2
from mapper_tda.metrics import compare_mapper_graphs, compute_graph_metrics, mapper_graph_to_networkx
from mapper_tda.pipeline import MapperConfig, build_mapper_graph, expand_configs_from_cli
from mapper_tda.preprocessing import preprocess_mapper_features


def synthetic_mapper_df(include_disc_facility: bool = True) -> pd.DataFrame:
    rows = 12
    data = {
        "pl_name": [f"planet_{idx}" for idx in range(rows)],
        "hostname": [f"star_{idx // 2}" for idx in range(rows)],
        "discoverymethod": ["Transit", "RV", "Transit", "RV"] * 3,
        "disc_year": [2010 + idx for idx in range(rows)],
        "sy_dist": np.linspace(10, 120, rows),
        "pl_rade": [0.9, 1.1, 1.4, 1.8, 2.2, 2.8, 3.5, 4.8, 6.0, 8.5, 12.5, 14.0],
        "pl_bmasse": [0.8, 1.2, 3.0, 5.5, 8.0, 12.0, 18.0, 45.0, 85.0, 150.0, 1100.0, 1400.0],
        "pl_dens": [5.3, 5.1, 4.8, 4.2, 3.4, 2.8, 2.1, 1.6, 1.4, 1.1, 0.9, 0.8],
        "pl_orbper": [2.0, 4.0, 8.0, 15.0, 40.0, 90.0, 180.0, 300.0, 500.0, 900.0, 1200.0, 1600.0],
        "pl_orbsmax": [0.03, 0.05, 0.08, 0.12, 0.25, 0.5, 0.8, 1.1, 1.6, 2.2, 3.0, 4.5],
        "pl_insol": [2500.0, 1700.0, 900.0, 500.0, 180.0, 90.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1],
        "pl_eqt": [1800.0, 1500.0, 1100.0, 850.0, 600.0, 420.0, 300.0, 240.0, 170.0, 130.0, 90.0, 60.0],
        "pl_orbeccen": [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22],
        "pl_rade_was_missing": [False, False, True, False, False, True, False, False, True, False, False, False],
        "pl_bmasse_was_missing": [False] * rows,
        "pl_dens_was_missing": [False, True, False, False, True, False, False, False, False, True, False, False],
        "pl_orbper_was_missing": [False] * rows,
        "pl_orbsmax_was_missing": [False, False, False, True, False, False, False, False, True, False, False, False],
        "pl_insol_was_missing": [False] * rows,
        "pl_eqt_was_missing": [False] * rows,
    }
    if include_disc_facility:
        data["disc_facility"] = ["ObsA"] * rows
    return pd.DataFrame(data)


class LensTests(unittest.TestCase):
    def test_make_lens_pca2_returns_two_columns(self) -> None:
        Z = np.arange(30, dtype=float).reshape(10, 3)
        lens, metadata = make_lens_pca2(Z)
        self.assertEqual(lens.shape, (10, 2))
        self.assertIn("explained_variance_ratio", metadata)

    def test_make_lens_density_returns_two_columns(self) -> None:
        Z = np.arange(30, dtype=float).reshape(10, 3)
        lens, metadata = make_lens_density(Z, k_density=4)
        self.assertEqual(lens.shape, (10, 2))
        self.assertEqual(metadata["k_density_requested"], 4)


class PreprocessingTests(unittest.TestCase):
    def test_estimate_dbscan_eps_is_positive(self) -> None:
        Z = np.linspace(0.0, 1.0, 40, dtype=float).reshape(20, 2)
        eps = estimate_dbscan_eps(Z, min_samples=4, percentile=90)
        self.assertGreater(eps, 0.0)

    def test_preprocess_mapper_features_prefers_original_columns_and_preserves_ids(self) -> None:
        df = pd.DataFrame(
            {
                "pl_name": ["a", "b", "c"],
                "hostname": ["h1", "h2", "h3"],
                "pl_rade": [10.0, 11.0, 12.0],
                "original_pl_rade": [1.0, 2.0, 4.0],
                "pl_bmasse": [20.0, 21.0, 22.0],
                "original_pl_bmasse": [2.0, 8.0, 32.0],
                "pl_dens": [30.0, 31.0, 32.0],
                "original_pl_dens": [4.0, 2.0, 1.0],
            }
        )
        work_df, Z, _, used_features = preprocess_mapper_features(
            df,
            features=["pl_rade", "pl_bmasse", "pl_dens"],
            log10_features=["pl_rade", "pl_bmasse", "pl_dens"],
            complete_case_only=True,
        )
        self.assertEqual(used_features, ["pl_rade", "pl_bmasse", "pl_dens"])
        self.assertEqual(Z.shape, (3, 3))
        self.assertListEqual(work_df["pl_rade"].tolist(), [1.0, 2.0, 4.0])
        self.assertIn("pl_name", work_df.columns)
        self.assertNotIn("pl_name", used_features)


class PipelineTests(unittest.TestCase):
    def test_build_mapper_graph_runs_on_small_dataframe(self) -> None:
        df = synthetic_mapper_df()
        result = build_mapper_graph(
            df,
            MapperConfig(space="joint", lens="pca2", n_cubes=4, overlap=0.30, min_samples=2, eps_percentile=80),
        )
        self.assertIn("graph", result)
        self.assertIn("node_table", result)
        self.assertIn("beta_0", result["graph_metrics"])
        self.assertIn("beta_1", result["graph_metrics"])

    def test_mapper_graph_to_networkx_generates_graph(self) -> None:
        graph = {
            "nodes": {"n1": [0, 1], "n2": [1, 2], "n3": [3]},
            "links": {"n1": ["n2"], "n2": ["n1", "n3"], "n3": ["n2"]},
        }
        nx_graph = mapper_graph_to_networkx(graph)
        self.assertIsInstance(nx_graph, nx.Graph)
        self.assertEqual(nx_graph.number_of_nodes(), 3)
        self.assertEqual(nx_graph.number_of_edges(), 2)

    def test_compute_graph_metrics_includes_betti_numbers(self) -> None:
        graph = {
            "nodes": {"n1": [0, 1], "n2": [1, 2], "n3": [3]},
            "links": {"n1": ["n2"], "n2": ["n1", "n3"], "n3": ["n2"]},
        }
        nx_graph = mapper_graph_to_networkx(graph)
        metrics = compute_graph_metrics(nx_graph, graph)
        self.assertIn("beta_0", metrics)
        self.assertIn("beta_1", metrics)

    def test_script_runs_without_optional_control_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "mapper_input.csv"
            reports_dir = tmp_path / "reports"
            synthetic_mapper_df(include_disc_facility=False).to_csv(csv_path, index=False)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "src" / "mapper_exodata.py"),
                    "--csv",
                    str(csv_path),
                    "--reports-dir",
                    str(reports_dir),
                    "--space",
                    "phys",
                    "--lens",
                    "pca2",
                    "--n-cubes",
                    "4",
                    "--overlap",
                    "0.30",
                    "--min-samples",
                    "2",
                    "--eps-percentile",
                    "80",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("Mapper/TDA generado correctamente", completed.stdout)
            self.assertTrue((reports_dir / "mapper_report.html").exists())

    def test_expand_configs_all_space_all_lens_contains_required_pairs(self) -> None:
        args = SimpleNamespace(
            space="all",
            lens="all",
            n_cubes=10,
            overlap=0.35,
            clusterer="dbscan",
            min_samples=4,
            eps_percentile=90.0,
            k_density=15,
            complete_case_only=True,
            include_eccentricity=False,
            random_state=42,
            grid=False,
        )
        configs = expand_configs_from_cli(args)
        config_names = {f"{config.space}_{config.lens}" for config in configs}
        expected = {
            "phys_pca2",
            "phys_density",
            "orb_pca2",
            "orb_density",
            "joint_pca2",
            "joint_density",
        }
        self.assertTrue(expected.issubset(config_names))

    def test_compare_mapper_graphs_returns_rows(self) -> None:
        metrics_df = pd.DataFrame(
            [
                {
                    "config_id": "phys_pca2_cubes10_overlap0p35",
                    "space": "phys",
                    "lens": "pca2",
                    "n_cubes": 10,
                    "overlap": 0.35,
                    "n_nodes": 8,
                    "n_edges": 10,
                    "beta_0": 1,
                    "beta_1": 3,
                    "graph_density": 0.35,
                    "average_degree": 2.5,
                    "average_clustering": 0.21,
                    "diameter_largest_component": 5,
                    "average_shortest_path_largest_component": 2.1,
                },
                {
                    "config_id": "orb_pca2_cubes10_overlap0p35",
                    "space": "orb",
                    "lens": "pca2",
                    "n_cubes": 10,
                    "overlap": 0.35,
                    "n_nodes": 10,
                    "n_edges": 12,
                    "beta_0": 1,
                    "beta_1": 3,
                    "graph_density": 0.26,
                    "average_degree": 2.4,
                    "average_clustering": 0.19,
                    "diameter_largest_component": 4,
                    "average_shortest_path_largest_component": 1.9,
                },
            ]
        )
        distances = compare_mapper_graphs(metrics_df)
        self.assertFalse(distances.empty)


if __name__ == "__main__":
    unittest.main()
