"""Tests for utils.py."""

import numpy as np

from multitask_personalization.utils import Bounded1DClassifier


def test_bounded_1d_classifier():
    """Tests for Bounded1DBayesianClassifier()."""

    model = Bounded1DClassifier(0.0, 1.0)
    X = [0.2, 0.4, 0.6, 0.8]
    Y = [False, True, True, False]
    model.fit_incremental(X, Y)
    assert np.isclose(model.predict_proba([0.1])[0], 0.0)
    assert np.isclose(model.predict_proba([0.3])[0], 0.5)
    assert np.isclose(model.predict_proba([0.5])[0], 1.0)
    assert np.isclose(model.predict_proba([0.7])[0], 0.5)
    assert np.isclose(model.predict_proba([0.9])[0], 0.0)
