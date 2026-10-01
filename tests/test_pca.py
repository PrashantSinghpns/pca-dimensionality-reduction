from src.run_pca import build_dataset, run_pca


def test_pca_returns_two_components_for_every_sample():
    features, _ = build_dataset()
    components, pca = run_pca(features)

    assert components.shape == (500, 2)
    assert 0 < pca.explained_variance_ratio_.sum() <= 1
