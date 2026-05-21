# Data

This directory contains the TableMatch benchmark files used by MTMatch.

## `Tablematch/`

Canonical benchmark view. Use this directory for most experiments.

Important files:

- `dataset_*.csv`: source tables
- `ground_truth_dataset1_beijing.csv`: two-table correspondence labels
- `multi_gt.csv`: multi-table positive equivalence edges
- `columns_*.txt` and `columns_mapping_train.csv`: auxiliary serialized files used by current training and evaluation code paths

## `TablematchMultiEM/`

Derived view for MultiEM-compatible evaluation.

Important files:

- `table_*.csv`: serialized column records with `tid`, `table_name`, `column_name`, and `sample_values`
- `ground_truth.txt`: semantic cluster memberships over `tid` values

## External EM Benchmarks

The record-level EM code expects task data under `data/em/<task-name>/` when used. Standard EM benchmark files are not bundled in this release unless you separately confirm redistribution rights.
