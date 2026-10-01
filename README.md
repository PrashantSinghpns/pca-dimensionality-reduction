# PCA Dimensionality Reduction

A reproducible demonstration of reducing five standardized features to two principal components with scikit-learn PCA.

## What it demonstrates

- Building a controlled five-feature dataset with three known groups.
- Standardizing input features before fitting PCA.
- Measuring the variance retained by the two-component projection.
- Producing a labelled CSV projection and a visual diagnostic.

This is a learning project built on synthetic data. The labels are only used to make the plot interpretable; PCA itself is unsupervised and does not use them while fitting.

## Project structure

```text
src/run_pca.py                 # Reproducible analysis pipeline
tests/test_pca.py              # Shape and explained-variance checks
notebooks/pca_exploration.ipynb # Original exploratory notebook
```

## Run locally

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python src/run_pca.py
pytest
```

The script creates `outputs/pca_projection.csv` and `outputs/pca_projection.png`.

## Key result

The exact variance retained is printed at runtime. It depends on the fixed random seed, so the result is reproducible.
