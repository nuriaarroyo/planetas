# Mapper Interpretive Summary

## Main findings
- Orbital Mapper shows high structural complexity with low imputation dependence.
- Thermal Mapper shows high complexity but high imputation dependence.
- Adding derived density slightly modifies, and in this run reduces, Mapper complexity.

## Promising signals
- The orbital PCA Mapper is a high-priority graph for scientific inspection: it combines nontrivial cycle structure with low imputation dependence.
- Highlighted node catalog available with 495 node-level interpretation records.
- Connected-component summaries available for 6 principal graphs.

## Risky configurations
- The thermal Mapper should be treated cautiously because more than half of its nodes exceed the high-imputation threshold.

## Density effect
Adding derived density reduces cycle complexity in the joint space, suggesting that density acts as a regularizing coordinate rather than introducing additional branching.

## Lens effect
Results are lens-sensitive; pca2 should remain the primary interpretation layer, while density/domain are sensitivity probes.

## Recommendations
- Bootstrap stability for principal graphs.
- Null models based on column-wise shuffling.
- Astrophysical review of highlighted nodes and components.
- Bootstrap was not run in this execution.
- Null models were not run in this execution.
- Multi-method Mapper comparison was skipped or partial because not all imputation inputs were available.
