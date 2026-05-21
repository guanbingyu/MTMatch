# GitHub Upload Guide

This directory is prepared as the repository root for a public GitHub release.

## Suggested Repository Name

`MTMatch`

## Before Uploading

Check these items:

- Confirm upstream redistribution rights for all raw tables in `data/Tablematch/`.
- Update `CITATION.cff` with the final paper venue, DOI, and GitHub URL after they are available.
- Update the repository URL in the GitHub description.
- Decide whether to create a GitHub release tag such as `v0.1.0`.

## Initial Upload

From inside `MTMatch_GitHub_release`:

```bash
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin https://github.com/<your-org-or-user>/MTMatch.git
git push -u origin main
```

## Suggested GitHub Description

`MTMatch: multi-granularity tabular matching with the TableMatch column-level benchmark.`

## Suggested Topics

`schema-matching`, `entity-matching`, `data-integration`, `table-matching`, `benchmark`, `roberta`
