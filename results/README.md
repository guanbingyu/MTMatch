# Result Snapshot

This directory stores compact result summaries that are useful when shipping the package with the paper.

## Included files

- `MTMatch_paper_result_snapshot.json`: headline MTMatch numbers quoted in the current manuscript.
- `MTMatch_em_ablation_4datasets_seed0.tsv`: four-dataset EM head ablation matrix used to support the methodology and experiment discussion.
- `em_stability_shared_full_3seeds/`: compact three-seed EM stability summaries.
- `framework_ablation/` and `framework_ablation_reference/`: compact ablation tables for the Tablematch framework variants.
- `framework_sensitivity/`: compact sensitivity tables for the current multi-table scoring settings.
- `tablematch_subset_enumeration/`: compact subset-enumeration summaries for the Tablematch benchmark.

These files are intended as lightweight reference outputs, not as substitutes for full reruns. Local command snapshots with absolute filesystem paths are excluded from this public package.
