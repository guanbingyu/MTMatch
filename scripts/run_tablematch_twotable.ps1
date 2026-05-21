$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

python MTMatch.py `
  --mode match `
  --reference data/Tablematch/dataset_1.csv `
  --target data/Tablematch/dataset_2-Beijing.csv `
  --lm roberta `
  --lm_only `
  --cluster_threshold 0.88
