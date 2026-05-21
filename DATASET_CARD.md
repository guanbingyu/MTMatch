# TableMatch Dataset Card

## Summary

TableMatch is a column-level benchmark for aligning semantically equivalent fields across heterogeneous structured tables in the air-quality and meteorological domain.

It supports two related tasks:

- two-table column correspondence prediction
- multi-table semantic cluster recovery across partially overlapping schemas

## Files

The canonical benchmark view is stored in `data/Tablematch/`.

| File | Description |
| --- | --- |
| `dataset_1.csv` | Beijing single-station PM2.5 reference table |
| `dataset_2-Beijing.csv` | Beijing multi-station table |
| `dataset_2-Chengdu.csv` | Chengdu multi-station table |
| `dataset_2-Guangzhou.csv` | Guangzhou multi-station table |
| `dataset_2-Shanghai.csv` | Shanghai multi-station table |
| `dataset_2-Shenyang.csv` | Shenyang multi-station table |
| `dataset_3.csv` | Italian air-quality table with a more heterogeneous schema |
| `dataset_4.csv` | Additional Beijing multi-station table from another source |
| `ground_truth_dataset1_beijing.csv` | Two-table correspondence labels between `dataset_1.csv` and `dataset_2-Beijing.csv` |
| `multi_gt.csv` | Multi-table positive cross-table equivalence edges |

The derived MultiEM-compatible view is stored in `data/TablematchMultiEM/`.

## Scale

- 8 canonical tables
- 16 two-table positive correspondences for `dataset_1.csv` versus `dataset_2-Beijing.csv`
- 482 labeled positive cross-table equivalence edges in `multi_gt.csv`
- 119 unique labeled columns participating in at least one multi-table edge
- 15 semantic clusters induced by the multi-table labels

## Recommended Evaluation Settings

- `TableMatch-homogeneous`: all tables except `dataset_3.csv`
- `TableMatch-full`: all 8 tables

Use `TableMatch-homogeneous` for same-domain comparison and `TableMatch-full` for the heterogeneous robustness setting.

## Provenance

The benchmark tables are adapted from public air-quality and meteorological sources cited in the manuscript:

- Beijing PM2.5 Data, UCI Machine Learning Repository
- PM2.5 Data of Five Chinese Cities, UCI Machine Learning Repository
- Air Quality, UCI Machine Learning Repository
- Beijing PM2.5 Data 2010-2015, iDataScience

The curated labels are benchmark-specific annotations created for TableMatch evaluation.

## Intended Use

TableMatch is suitable for schema matching, table integration, column representation learning, multi-table semantic clustering, and robustness studies on partially overlapping schemas.

It is not intended as a general-purpose time-series forecasting benchmark, a causal environmental analysis benchmark, or a universal schema-matching benchmark across arbitrary domains.

## Privacy And Sensitive Data

The benchmark appears to contain environmental monitoring data and schema metadata rather than personal data. Even so, verify the redistribution terms of each upstream source before public release.

## Release Note

If any upstream source cannot be redistributed directly, publish the curated labels and reconstruction instructions instead of mirroring the raw tables.
