# Clinical Infection Prediction with Attention-Based Tabular Deep Learning

## Project Overview

This research repository explores prediction of clinical infection from structured cerebrospinal-fluid, blood-test, demographic, and clinical variables. The portfolio centers on an attention-based tabular neural network (AMFormer) and preserves classical baselines, hyperparameter searches, interpretation outputs, and historical experiments.

The code and results are research artifacts, not a clinically validated diagnostic system.

## Research Motivation

Clinical infection prediction is an imbalanced binary-classification problem in which false negatives may be especially consequential. This project investigates whether feature-wise attention and learned feature interactions can complement conventional machine-learning baselines while retaining interpretable summaries.

## Dataset

The original README described 915 records and 20 model features. Inputs include cerebrospinal-fluid measurements, blood biomarkers, demographic variables, and clinical observations. Existing spreadsheets and derived tables may be patient-level clinical data. Their provenance, consent, de-identification, and permission for public distribution have not been independently verified.

See [`data/README.md`](data/README.md) before using or publishing any dataset. Do not commit new raw patient-level data.

## Methodology

The primary workflow consists of rule-based cleaning and encoding, stratified five-fold cross-validation, fold-local preprocessing and imbalance handling, model training, and export of metrics, predictions, attention matrices, and feature-importance summaries.

### AMFormer

The project contains attention-based tabular architectures that embed features, learn cross-feature interactions, and use multi-head or sparse attention before binary classification. Portfolio entry points preserve the original model and training logic; older variants remain under `experiments/archive/`.

### Class Imbalance

The main experiments use combinations of SMOTE, positive-class weighting, focal loss, and fixed decision thresholds. These operations must remain fold-local to avoid leakage.

### Cross-Validation

Reported experiments use five-fold stratified cross-validation. Fold-level outputs are retained so aggregate performance can be audited rather than inferred from one split.

### Baseline Comparison

`baselines/` preserves logistic regression, decision tree, random forest, gradient boosting, XGBoost, and FT-Transformer experiments. The clearest five-fold classical entry point is `baselines/train_baselines.py`; ambiguous FT-Transformer iterations retain their original names in the archive.

### Interpretation

`src/analysis/feature_importance.py` computes permutation-based feature importance. `src/analysis/attention_analysis.py` summarizes a saved attention matrix. These are exploratory explanations, not proof of causality or clinical relevance.

## Preliminary Performance

The original project summary reported:

| Metric | Reported value |
| --- | ---: |
| Accuracy | 80.3% |
| ROC-AUC | 0.824 |
| Specificity | 92.3% |
| Sensitivity | 50.9% |

These are preliminary repository-reported results. They have not been externally reproduced, prospectively validated, or clinically validated. The relatively low reported sensitivity is particularly important for the intended use case.

## Repository Structure

```text
infection/
├── src/
│   ├── models/amformer.py
│   ├── training/train_amformer.py
│   ├── preprocessing/data_transformer.py
│   └── analysis/
├── baselines/
├── experiments/
│   ├── hyperparameter_search/
│   └── archive/
├── results/
├── checkpoints/
└── data/
```

Historical experiments are preserved in the archive rather than silently deleted or rewritten.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

Run commands from the repository root.

```bash
python -m src.preprocessing.data_transformer
python -m src.training.train_amformer              # computationally expensive
python baselines/train_baselines.py
python -m src.analysis.attention_analysis
python experiments/hyperparameter_search/search_amformer.py  # expensive
```

The current scripts read `data/legacy/original.xlsx` to preserve compatibility. For public distribution, use only a properly authorized dataset or adapt the configured path.

## Reproducibility Notes

- Seeds are set in primary experiments, but GPU kernels and library versions can introduce nondeterminism.
- Preprocessing, sampling, and scaling should remain fold-local.
- Existing checkpoints/results record historical runs and may differ across hardware or dependencies.
- No expensive training is part of repository validation.
- Consult `experiments/archive/README.md` before comparing exploratory variants.

## Technologies

Python, pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, PyTorch, Matplotlib, Seaborn, and OpenPyXL.

## Limitations

- Small, apparently single-source dataset.
- Unverified data governance and public-release status.
- Class imbalance and modest reported sensitivity.
- No external, temporal, or prospective validation.
- Possible selection bias, missingness, and preprocessing assumptions.
- Attention and feature importance do not establish causal effects.

## Future Work

- Verify data governance and publish only an approved de-identified or synthetic dataset.
- Add schema-validated configuration and automated tests.
- Reproduce results from a clean environment with locked dependencies.
- Evaluate calibration, uncertainty, fairness, and clinically relevant operating points.
- Perform external and prospective validation against predefined protocols.

## Clinical-Use Disclaimer

This repository is for research and educational use only. It must not be used to diagnose, treat, triage, or make clinical decisions. Clinical translation requires governance, privacy review, regulatory assessment, independent validation, and qualified clinical oversight.
