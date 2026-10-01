# PCA Dimensionality Reduction

[![Python checks](https://github.com/PrashantSinghpns/pca-dimensionality-reduction/actions/workflows/ci.yml/badge.svg)](https://github.com/PrashantSinghpns/pca-dimensionality-reduction/actions/workflows/ci.yml)

A reproducible five-feature to two-component projection with a reusable scikit-learn preprocessing pipeline, variance diagnostics, component weights, and reconstruction error.

## Run the experiment

Use Python 3.11 or 3.12. These experiments generate their own synthetic data and need no external download.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt pytest
python src/run_pca.py --output-dir outputs
python -m pytest -q
```

For Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`.

## Method and interpretation

StandardScaler fits five features from 500 synthetic points generated with three blob centers and seed 42. A two-component PCA is fitted without using the group labels. Labels color the visualization only. The exported pipeline preserves the fitted scaling and projection; call its `transform` method for new rows without refitting. Reconstruction MSE is measured in original feature units on the demonstration data and is descriptive, not a held-out prediction score.

## Generated evidence

| Output under `outputs/` | Purpose |
|---|---|
| `pca_projection.csv`, `pca_projection.png` | Two-dimensional projection and group visualization |
| `explained_variance.csv` | Full and cumulative explained variance |
| `component_weights.csv` | Principal-axis weights in standardized feature coordinates |
| `metrics.json` | Retained variance, reconstruction MSE, data specification, package versions |
| `projection.joblib` | Fitted StandardScaler/PCA pipeline for new samples |

The command prints the measured report. Package versions are recorded because exact numerical values can vary across environments. Generated artifacts are ignored by git and can be recreated from the command above.

## Repository map

| Path | Purpose |
|---|---|
| `src/` | Runnable experiment and reusable functions |
| `tests/` | Shape, index, new-sample, or noise-handling checks |
| `notebooks/` | Original exploratory notebook; the script is the reproducible entry point |
| `.github/workflows/ci.yml` | Syntax and unit checks on Python 3.11/3.12 |

## Limitations

PCA is linear and sensitive to scaling and outliers. Retaining variance does not prove downstream predictive usefulness. Component signs can differ across numerical environments without changing the projection geometry.

Synthetic results demonstrate algorithm behavior; they do not establish performance on a business dataset. Apply held-out preprocessing when embedding these methods in a supervised workflow.


## Reference experiment

A locally executed synthetic reference run is recorded in [docs/reference_metrics.json](docs/reference_metrics.json), including the actual package versions. It is a reproducibility example, not a production benchmark.

Two components retained **85.82%** of variance on 500 generated samples. Reconstruction MSE was **2.0615** in the original feature units.
