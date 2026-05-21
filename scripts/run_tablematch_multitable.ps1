$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

python MTMatch.py `
  --mode multi_match `
  --multi_tables_dir data/Tablematch `
  --multi_tables_glob "dataset_*.csv" `
  --multi_top_k 2 `
  --lm roberta `
  --lm_only `
  --cluster_threshold 0.88 `
  --multi_embedding_weight 0.85 `
  --multi_profile_weight 0.05 `
  --multi_name_weight 0.00 `
  --multi_semantic_family_prior `
  --multi_expand_edges_from_clusters `
  --multi_ground_truth data/Tablematch/multi_gt.csv
