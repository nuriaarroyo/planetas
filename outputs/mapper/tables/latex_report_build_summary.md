# LaTeX Report Build Summary

Generated report assets for `latex/03_mapper/` from measured CSV, JSON, and markdown outputs.

## Inputs Used

- `outputs/mapper/tables/output_manifest.csv`
- `outputs/mapper/tables/output_consistency_warnings.md`
- `outputs/mapper/tables/mapper_graph_metrics_all_existing.csv`
- `outputs/mapper/tables/mapper_lens_sensitivity_all_existing.csv`
- `outputs/mapper/tables/mapper_space_comparison_all_existing.csv`
- `outputs/mapper/tables/main_graph_selection.csv`
- `outputs/mapper/tables/discoverymethod_permutation_null.csv`
- `outputs/mapper/tables/discoverymethod_enrichment_summary.csv`
- `outputs/mapper/tables/component_discovery_bias.csv`
- `outputs/mapper/tables/final_region_synthesis.csv`
- `reports/imputation/physical_derivations.csv`
- `reports/imputation/outputs/tables/mapper_coverage_summary.csv`

Permutation n_perm detected: 1000

## Tables Generated

- `latex/03_mapper/tables/table_all_existing_mapper_metrics.tex`
- `latex/03_mapper/tables/table_bias_permutation_null.tex`
- `latex/03_mapper/tables/table_component_discovery_bias.tex`
- `latex/03_mapper/tables/table_final_region_synthesis.tex`
- `latex/03_mapper/tables/table_lens_sensitivity.tex`
- `latex/03_mapper/tables/table_main_graph_selection.tex`
- `latex/03_mapper/tables/table_output_manifest_summary.tex`
- `latex/03_mapper/tables/table_space_comparison.tex`
- `latex/03_mapper/tables/table_top_enriched_nodes.tex`

## Sections Generated

- `latex/03_mapper/sections/00_abstract.tex`
- `latex/03_mapper/sections/01_introduction.tex`
- `latex/03_mapper/sections/02_astrophysical_context.tex`
- `latex/03_mapper/sections/03_data_and_imputation.tex`
- `latex/03_mapper/sections/04_mapper_methodology.tex`
- `latex/03_mapper/sections/05_output_reconciliation.tex`
- `latex/03_mapper/sections/06_mapper_results.tex`
- `latex/03_mapper/sections/07_selected_graphs.tex`
- `latex/03_mapper/sections/08_astrophysical_interpretation.tex`
- `latex/03_mapper/sections/09_observational_bias_audit.tex`
- `latex/03_mapper/sections/10_permutation_null_test.tex`
- `latex/03_mapper/sections/11_region_synthesis.tex`
- `latex/03_mapper/sections/12_limitations.tex`
- `latex/03_mapper/sections/13_conclusion.tex`

## Figures Copied/Updated

- No figure files needed copying; existing figure files were reused.

## Main TeX

- `latex/03_mapper/mapper_report.tex`

## Notes

- Mapper was not rerun.
- Feature-space definitions were not modified.
- Tables omit optional columns only when the source CSV does not contain them.
- The report frames Mapper as exploratory evidence, not as a final exoplanet taxonomy.
