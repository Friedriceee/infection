# Data placement and privacy

This project expects a clinical table with an `outcome` target and the feature columns referenced by the preprocessing and training scripts.

## Privacy warning

Files preserved under `data/legacy/` appear, from their names, schema, and the original README, to contain record-level clinical observations. They were moved without duplication or deletion. Their consent, de-identification, ownership, and authorization for public distribution have not been verified.

Before publication, the owner must review every tracked file under `data/legacy/` and the historical archives. If distribution is not explicitly permitted, remove the data from the branch **and Git history** through an approved process, and assess whether derived artifacts are also sensitive.

Do not print, publish, or upload record contents during review.

Recommended local placement:

```text
data/
├── raw/          # authorized source data; ignored by Git
├── processed/    # reproducible derived data
└── README.md
```

Portfolio scripts currently reference `data/legacy/original.xlsx` to preserve behavior. For a public release, point them to an approved local file or provide documented synthetic data.
