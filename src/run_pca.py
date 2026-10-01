"""Fit, inspect, and persist a standardized PCA projection on synthetic data."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

def build_dataset():
    features, labels = make_blobs(n_samples=500, centers=3, n_features=5,
                                  cluster_std=1.5, random_state=RANDOM_STATE)
    return pd.DataFrame(features, columns=[f"feature_{i}" for i in range(1, 6)]), pd.Series(labels, name="label")

def fit_projection(features):
    """Keep scaling and PCA together so future samples reuse the fitted transform."""
    if min(features.shape) < 2:
        raise ValueError("PCA needs at least two samples and two features.")
    pipeline = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=2, random_state=RANDOM_STATE))])
    components = pipeline.fit_transform(features)
    return pd.DataFrame(components, columns=["PC1", "PC2"], index=features.index), pipeline

def run_pca(features):
    """Compatibility interface returning the projection and fitted PCA estimator."""
    components, pipeline = fit_projection(features)
    return components, pipeline.named_steps["pca"]

def save_outputs(results, pca, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "pca_projection.csv", index=False)
    explained = float(pca.explained_variance_ratio_.sum())
    fig, axis = plt.subplots(figsize=(8, 6))
    for label, group in results.groupby("label"):
        axis.scatter(group.PC1, group.PC2, alpha=0.75, label=f"Group {label}")
    axis.set(title=f"PCA projection (variance retained: {explained:.1%})", xlabel="PC1", ylabel="PC2")
    axis.legend(title="Synthetic group")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_projection.png", dpi=160)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    features, labels = build_dataset()
    components, pipeline = fit_projection(features)
    pca = pipeline.named_steps["pca"]
    save_outputs(components.assign(label=labels), pca, args.output_dir)
    full_pca = PCA().fit(pipeline.named_steps["scaler"].transform(features))
    variances = full_pca.explained_variance_ratio_
    pd.DataFrame({"component": range(1, len(variances) + 1), "explained_variance_ratio": variances,
                  "cumulative_variance": np.cumsum(variances)}).to_csv(args.output_dir / "explained_variance.csv", index=False)
    pd.DataFrame(pca.components_.T, index=features.columns, columns=["PC1_weight", "PC2_weight"]).to_csv(args.output_dir / "component_weights.csv", index_label="feature")
    reconstructed = pipeline.inverse_transform(components)
    report = {"dataset": "synthetic_blobs", "rows": len(features), "input_features": list(features.columns),
              "random_state": RANDOM_STATE, "retained_components": 2,
              "variance_retained": float(pca.explained_variance_ratio_.sum()),
              "reconstruction_mse_original_units": float(np.mean((features.to_numpy() - reconstructed) ** 2)),
              "versions": {n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scikit-learn", "matplotlib"]}}
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    joblib.dump(pipeline, args.output_dir / "projection.joblib")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
