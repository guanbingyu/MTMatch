# Reproducibility Notes

This document gives the shortest path for checking that the public package runs.

## Environment

Install PyTorch for your system first, then run:

```bash
pip install -r requirements.txt
```

## Smoke Tests

From the repository root, run:

```bash
python MTMatch.py --help
python MTMatch_train.py --help
```

Then run one TableMatch example:

```bash
python MTMatch.py --mode match --reference data/Tablematch/dataset_1.csv --target data/Tablematch/dataset_2-Beijing.csv --lm roberta --lm_only --cluster_threshold 0.88
```

## Paper-Oriented Runs

Pairwise TableMatch run:

```bash
python MTMatch.py --mode pairwise_match --pairwise_tables_dir data/Tablematch --pairwise_tables_glob "dataset_*.csv" --lm roberta --lm_only --cluster_threshold 0.88 --multi_ground_truth data/Tablematch/multi_gt.csv
```

Multi-table TableMatch run:

```bash
python MTMatch.py --mode multi_match --multi_tables_dir data/Tablematch --multi_tables_glob "dataset_*.csv" --multi_top_k 2 --lm roberta --lm_only --cluster_threshold 0.88 --multi_embedding_weight 0.85 --multi_profile_weight 0.05 --multi_name_weight 0.00 --multi_semantic_family_prior --multi_expand_edges_from_clusters --multi_ground_truth data/Tablematch/multi_gt.csv
```

## Outputs

Runtime outputs are written under `outputs/` by default. The `.gitignore` excludes generated logs, checkpoints, MLflow tracking data, and Python caches.

## Result Snapshots

Compact result tables are provided under `results/`. They are intended as reference summaries, not as substitutes for rerunning the experiments.
