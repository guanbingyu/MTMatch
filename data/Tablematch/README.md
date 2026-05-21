# TableMatch Canonical Data

This directory is the canonical benchmark view for `TableMatch`.

## Included tables

- `dataset_1.csv`
- `dataset_2-Beijing.csv`
- `dataset_2-Chengdu.csv`
- `dataset_2-Guangzhou.csv`
- `dataset_2-Shanghai.csv`
- `dataset_2-Shenyang.csv`
- `dataset_3.csv`
- `dataset_4.csv`

These are the files used by the current TableMatch experiments and manuscript.

## Core label files

### `ground_truth_dataset1_beijing.csv`

Purpose:

- two-table column correspondence labels between `dataset_1.csv` and `dataset_2-Beijing.csv`

Schema:

- `reference_col`
- `target_col`

Each row denotes one positive correspondence.

### `multi_gt.csv`

Purpose:

- multi-table positive cross-table equivalence edges

Schema:

- `src_table`
- `src_col`
- `dst_table`
- `dst_col`

Each row denotes one positive equivalence edge between two columns from different tables. These edges induce the benchmark's global semantic clusters.

## Auxiliary training files

### `columns_mapping_train.csv`

Purpose:

- compact labeled pairs for existing pairwise wrappers and sanity checks

Schema:

- `left_col`
- `right_col`
- `label`

### `columns_train.txt`, `columns_valid.txt`, `columns_test.txt`

Purpose:

- serialized pairwise inputs used by the current code paths

### `sudowoodo_pretrain_data/`

Purpose:

- auxiliary serialized inputs used by the current Sudowoodo-style or shared-backbone experiments

These auxiliary files are useful for reproducing the current workspace experiments, but they are not the primary definition of the benchmark. The benchmark itself is the set of canonical tables plus the correspondence labels.

## Recommended split names

- `TableMatch-homogeneous`: all tables except `dataset_3.csv`
- `TableMatch-full`: all 8 tables

## Notes for public release

- Keep this directory as the canonical benchmark source of truth.
- If redistribution rights for some raw tables are limited, keep the labels here and provide reconstruction scripts for the raw tables instead.
- Preserve the file names if you want to keep existing experiment wrappers working without code changes.
