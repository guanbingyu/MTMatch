# MTMatch

MTMatch is a research prototype for multi-granularity tabular matching. It supports column-level table matching on the TableMatch benchmark and record-level entity matching through a shared RoBERTa-based backbone.

This repository is prepared as a public artifact for the MTMatch paper. It contains the cleaned source code, the TableMatch benchmark files, lightweight result summaries, and reproducibility notes. Large training checkpoints, MLflow runs, local logs, manuscript build files, and unrelated baseline repositories are intentionally excluded.

## What Is Included

- `MTMatch.py`: column-level matching entry point for two-table, pairwise, and multi-table settings.
- `MTMatch_train.py`: record-level entity matching training entry point.
- `selfsl/`: shared representation learning, augmentation, and EM utilities.
- `data/Tablematch/`: canonical TableMatch benchmark tables and labels.
- `data/TablematchMultiEM/`: derived MultiEM-compatible view of TableMatch.
- `results/`: compact result snapshots and ablation summaries.
- `docs/`: release, dataset, and reproducibility notes.
- `scripts/`: small convenience wrappers for common TableMatch runs.

## Installation

Create a Python environment, install PyTorch for your CUDA or CPU setup, and then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The code was developed with Python 3.10+ style environments. If your GPU requires a specific PyTorch wheel, install that first from the official PyTorch instructions, then run the command above.

## Quick Start

Run a two-table column matching example:

```bash
python MTMatch.py --mode match --reference data/Tablematch/dataset_1.csv --target data/Tablematch/dataset_2-Beijing.csv --lm roberta --lm_only --cluster_threshold 0.88
```

Run pairwise evaluation across the TableMatch benchmark:

```bash
python MTMatch.py --mode pairwise_match --pairwise_tables_dir data/Tablematch --pairwise_tables_glob "dataset_*.csv" --lm roberta --lm_only --cluster_threshold 0.88 --multi_ground_truth data/Tablematch/multi_gt.csv
```

Run the multi-table configuration used for the paper:

```bash
python MTMatch.py --mode multi_match --multi_tables_dir data/Tablematch --multi_tables_glob "dataset_*.csv" --multi_top_k 2 --lm roberta --lm_only --cluster_threshold 0.88 --multi_embedding_weight 0.85 --multi_profile_weight 0.05 --multi_name_weight 0.00 --multi_semantic_family_prior --multi_expand_edges_from_clusters --multi_ground_truth data/Tablematch/multi_gt.csv
```

Record-level EM code is included, but the public release currently focuses on TableMatch. If you have redistribution rights for standard EM benchmarks, place them under `data/em/<task-name>/` following the expected `train.txt`, `valid.txt`, `test.txt`, `tableA.txt`, and `tableB.txt` layout.

## Dataset

TableMatch is an air-quality and meteorological benchmark for column-level table matching. It includes 8 tables, two-table correspondence labels, and multi-table N:M semantic-cluster labels.

See [DATASET_CARD.md](./DATASET_CARD.md) and [data/README.md](./data/README.md) for details.

## Reproducibility

For a compact reproduction guide, see [docs/REPRODUCIBILITY.md](./docs/REPRODUCIBILITY.md). Lightweight result snapshots are under [results/](./results/).

## Citation

If you use this code or dataset, please cite the paper when it becomes available. A placeholder citation metadata file is provided in [CITATION.cff](./CITATION.cff) and should be updated with the final venue, DOI, and repository URL.

## License And Data Terms

Code in this release includes components derived from Sudowoodo and retains the bundled BSD-3-Clause license. See [LICENSE](./LICENSE) and [NOTICE.md](./NOTICE.md).

The curated TableMatch annotations are newly prepared for this benchmark. Upstream raw data tables retain the terms of their original sources. Before making the GitHub repository public, verify redistribution rights for every upstream source table listed in [DATASET_CARD.md](./DATASET_CARD.md).
