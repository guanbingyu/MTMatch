$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

python MTMatch.py `
  --mode pairwise_match `
  --pairwise_tables_dir data/Tablematch `
  --pairwise_tables_glob "dataset_*.csv" `
  --lm roberta `
  --lm_only `
  --cluster_threshold 0.88 `
  --multi_ground_truth data/Tablematch/multi_gt.csv
