# Graph Render Manifest

Clean static Mapper figures were generated from existing graph/node/edge/config outputs only.

## Render Settings

- Requested layout: `auto`
- Static backend: `matplotlib` + `networkx`
- Node sizing: sqrt membership scaling, capped to 20-220 points for static figures
- Edge rendering: light gray, alpha 0.5, width 0.8, drawn before nodes
- Mapper was not rerun.
- `feature_sets.py` was not modified.

## Graphs Rendered

| config_id | title | nodes | edges | layout_used |
| --- | --- | ---: | ---: | --- |
| `phys_min_pca2_cubes10_overlap0p35` | Physical / mass–radius | 66 | 105 | `lens_1_mean/lens_2_mean` |
| `phys_density_pca2_cubes10_overlap0p35` | Physical + density | 67 | 102 | `lens_1_mean/lens_2_mean` |
| `orbital_pca2_cubes10_overlap0p35` | Orbital | 124 | 196 | `lens_1_mean/lens_2_mean` |
| `joint_no_density_pca2_cubes10_overlap0p35` | Joint / no density | 62 | 134 | `lens_1_mean/lens_2_mean` |
| `joint_pca2_cubes10_overlap0p35` | Joint + density | 56 | 126 | `lens_1_mean/lens_2_mean` |
| `thermal_pca2_cubes10_overlap0p35` | Thermal | 121 | 150 | `lens_1_mean/lens_2_mean` |

## Static Outputs

- `latex/03_mapper/figures/main_graphs_by_evidence_class.pdf`
- `latex/03_mapper/figures/main_graphs_by_evidence_class.png`
- `latex/03_mapper/figures/orbital_graph_evidence_class.pdf`
- `latex/03_mapper/figures/orbital_graph_evidence_class.png`
- `latex/03_mapper/figures/orbital_graph_discoverymethod.pdf`
- `latex/03_mapper/figures/orbital_graph_discoverymethod.png`
- `latex/03_mapper/figures/orbital_graph_imputation.pdf`
- `latex/03_mapper/figures/orbital_graph_imputation.png`
- `latex/03_mapper/figures/region_class_counts.pdf`
- `latex/03_mapper/figures/region_class_counts.png`
- `latex/03_mapper/figures/astro_main_graphs_by_candidate_population.pdf`
- `latex/03_mapper/figures/astro_main_graphs_by_candidate_population.png`
- `latex/03_mapper/figures/astro_main_graphs_by_radius_class.pdf`
- `latex/03_mapper/figures/astro_main_graphs_by_radius_class.png`
- `latex/03_mapper/figures/astro_main_graphs_by_orbit_class.pdf`
- `latex/03_mapper/figures/astro_main_graphs_by_orbit_class.png`
- `latex/03_mapper/figures/astro_main_graphs_by_imputation_fraction.pdf`
- `latex/03_mapper/figures/astro_main_graphs_by_imputation_fraction.png`
- `latex/03_mapper/figures/astro_orbital_graph_by_orbit_class.pdf`
- `latex/03_mapper/figures/astro_orbital_graph_by_orbit_class.png`
- `latex/03_mapper/figures/astro_joint_no_density_graph_by_candidate_population.pdf`
- `latex/03_mapper/figures/astro_joint_no_density_graph_by_candidate_population.png`
- `latex/03_mapper/figures/astro_thermal_graph_by_thermal_class.pdf`
- `latex/03_mapper/figures/astro_thermal_graph_by_thermal_class.png`

## Interactive Outputs

- `outputs/mapper/interactive/main_graphs_by_evidence_class.html`
- `outputs/mapper/interactive/orbital_graph_evidence_class.html`
- `outputs/mapper/interactive/orbital_graph_discoverymethod.html`
- `outputs/mapper/interactive/orbital_graph_imputation.html`
- `outputs/mapper/interactive/astro_main_graphs_by_candidate_population.html`
- `outputs/mapper/interactive/astro_main_graphs_by_radius_class.html`
- `outputs/mapper/interactive/astro_main_graphs_by_orbit_class.html`
- `outputs/mapper/interactive/astro_main_graphs_by_imputation_fraction.html`
- `outputs/mapper/interactive/astro_orbital_graph_by_orbit_class.html`
- `outputs/mapper/interactive/astro_joint_no_density_graph_by_candidate_population.html`
- `outputs/mapper/interactive/astro_thermal_graph_by_thermal_class.html`

## Inputs

- `outputs/mapper/graphs/`
- `outputs/mapper/nodes/`
- `outputs/mapper/edges/`
- `outputs/mapper/tables/final_region_synthesis.csv`
- `outputs/mapper/tables/node_discovery_bias.csv`
- `outputs/mapper/tables/discoverymethod_enrichment_summary.csv`
