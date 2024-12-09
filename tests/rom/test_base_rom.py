"""Tests for base ROM models."""

import numpy as np

from multitask_personalization.rom.models import (
    ot_angles_to_pybullet_angles,
    pybullet_angles_to_ot_angles,
)
from multitask_personalization.utils import (
    DIMENSION_LIMITS,
    DIMENSION_NAMES,
)


def test_pybullet_ot_angle_conversion():
    """Tests for pybullet_angles_to_ot_angles() and inverse."""
    rng = np.random.default_rng(123)
    lower = [DIMENSION_LIMITS[n][0] for n in DIMENSION_NAMES]
    upper = [DIMENSION_LIMITS[n][1] for n in DIMENSION_NAMES]
    for _ in range(100):
        ot_angle = rng.uniform(lower, upper)
        pb_angle = ot_angles_to_pybullet_angles(ot_angle)
        recovered_ot_angle = pybullet_angles_to_ot_angles(pb_angle)
        assert np.allclose(ot_angle % 180, recovered_ot_angle % 180)
