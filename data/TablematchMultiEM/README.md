# TableMatch MultiEM View

This directory stores a transformed view of `TableMatch` for the `MultiEM` pipeline.

## Files

### `table_*.csv`

Each row represents one benchmark column serialized as a lightweight record.

Schema:

- `tid`: global integer identifier used by the MultiEM ground truth
- `table_name`: original source table name from the canonical benchmark
- `column_name`: source column header
- `sample_values`: compact textual serialization of sampled cell values

### `ground_truth.txt`

Each non-empty line is one semantic cluster. A line contains comma-separated `tid` values, for example:

```text
18,36,53,70,87,119
```

That means all listed `tid` values belong to the same semantic concept cluster.

## Relationship to the canonical benchmark

- canonical benchmark tables and labels live in `../Tablematch`
- this directory is a derived representation for MultiEM-compatible evaluation

If you publish both views, document that `TablematchMultiEM` is derived from the canonical `TableMatch` files and should not be treated as a separate benchmark.
