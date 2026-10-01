import numpy as np
from src.run_pca import build_dataset, fit_projection

def test_saved_transform_applies_to_new_rows_without_refitting():
    features, _ = build_dataset()
    features.index = range(1000, 1500)
    components, pipeline = fit_projection(features.iloc[:400])
    assert components.index.equals(features.index[:400])
    means = pipeline.named_steps["scaler"].mean_.copy()
    projected = pipeline.transform(features.iloc[400:])
    assert projected.shape == (100, 2)
    np.testing.assert_array_equal(means, pipeline.named_steps["scaler"].mean_)
    assert pipeline.inverse_transform(projected).shape == (100, 5)
