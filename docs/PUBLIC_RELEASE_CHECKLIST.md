# TableMatch Public Release Checklist

## 1. Scope and packaging

- Decide whether the public artifact is `benchmark only` or `benchmark + full reproducibility package`.
- Keep one canonical archive root name, preferably `TableMatch`.
- Decide whether legacy paths such as `Tablematch` remain on disk or are renamed before release.
- Freeze a release version, release date, and checksum manifest.

## 2. Legal and provenance checks

- Verify redistribution terms for each raw source dataset.
- Verify whether the iDataScience-derived Beijing multi-station table may be mirrored directly.
- Add a `LICENSE` or `TERMS` file for the benchmark package.
- Distinguish clearly between upstream raw data and newly curated labels.
- Add citation links or BibTeX entries for all upstream sources.

## 3. Documentation

- Keep the root `README.md` as the front door for external users.
- Keep `DATASET_CARD.md` aligned with the actual file contents and counts.
- Document the exact semantics of `ground_truth_dataset1_beijing.csv`.
- Document that `multi_gt.csv` stores positive cross-table equivalence edges.
- Document that `TablematchMultiEM/ground_truth.txt` stores cluster membership over `tid` values.
- State the recommended evaluation splits: `TableMatch-homogeneous` and `TableMatch-full`.

## 4. Release hygiene

- Exclude caches, logs, temporary outputs, and training checkpoints from the release archive.
- Remove manuscript build byproducts unless the manuscript source release explicitly needs them.
- Check that no absolute local filesystem paths remain in shared metadata or result snapshots.
- Check for accidental binary bloat in `tmp`, `output`, and experiment artifact folders.

## 5. Reproducibility

- Smoke-test the main scripts under `code/scripts`.
- Verify at least one two-table run, one pairwise folder run, and one multi-table run.
- Verify that data paths in the scripts resolve against the packaged directory layout.
- Confirm that the release archive includes only files required by those smoke tests.

## 6. Metadata and citation

- Add a canonical citation file if you want GitHub or Zenodo-friendly metadata.
- Add the benchmark description sentence to the release page:
- `TableMatch is an air-quality and meteorological benchmark with two-table correspondence labels and multi-table N:M semantic-cluster labels for column-level matching.`
- If a paper DOI exists, add it to the root docs and dataset card.

## 7. Fallback plan if redistribution is restricted

- Publish label files, schema documentation, and reconstruction scripts only.
- Replace raw tables with download instructions and checksums.
- Keep the same label semantics and file names so existing evaluation code still works after reconstruction.
