"""Reproducible PCA demonstration using a labelled synthetic dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def build_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Create a five-feature dataset with three known groups for demonstration."""
    features, labels = make_blobs(
        n_samples=500,
        centers=3,
        n_features=5,
        cluster_std=1.5,
        random_state=RANDOM_STATE,
    )
    columns = [f"feature_{index}" for index in range(1, 6)]
    return pd.DataFrame(features, columns=columns), pd.Series(labels, name="label")


def run_pca(features: pd.DataFrame) -> tuple[pd.DataFrame, PCA]:
    """Scale features, then retain two principal components."""
    scaled_features = StandardScaler().fit_transform(features)  # Standardize before PCA.
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(scaled_features)
    return pd.DataFrame(components, columns=["PC1", "PC2"]), pca


def save_outputs(results: pd.DataFrame, pca: PCA, output_dir: Path) -> None:
    """Save tabular results and a labelled projection plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "pca_projection.csv", index=False)

    explained = pca.explained_variance_ratio_.sum()
    fig, axis = plt.subplots(figsize=(8, 6))
    for label, group in results.groupby("label"):
        axis.scatter(group["PC1"], group["PC2"], alpha=0.75, label=f"Group {label}")
    axis.set(title=f"PCA projection (explained variance: {explained:.1%})", xlabel="PC1", ylabel="PC2")
    axis.legend(title="Known group")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_projection.png", dpi=160)
    plt.close(fig)


def main() -> None:
    features, labels = build_dataset()
    components, pca = run_pca(features)
    results = components.assign(label=labels)
    save_outputs(results, pca, Path("outputs"))
    print(f"Two components retain {pca.explained_variance_ratio_.sum():.1%} of variance.")


if __name__ == "__main__":
    main()
