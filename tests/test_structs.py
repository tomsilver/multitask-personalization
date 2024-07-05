"""Tests for structs.py."""

import numpy as np

from multitask_personalization.structs import CategoricalDistribution


def test_categorical_distribution():
    """Tests for CategoricalDistribution()."""
    dist = CategoricalDistribution({0: 25, 1: 25, 2: 50}, normalize=True)
    assert np.isclose(dist[0], 0.25)
    rng = np.random.default_rng(123)
    item = dist.sample(rng)
    assert item in {0, 1, 2}
    other_dist = CategoricalDistribution({0: 0.25, 1: 0.25, 2: 0.50})
    assert dist == other_dist
    dist_set = {dist, other_dist}
    assert len(dist_set) == 1
