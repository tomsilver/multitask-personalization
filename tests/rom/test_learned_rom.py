"""Tests for rom/models.py."""

import numpy as np

from multitask_personalization.envs.pybullet.pybullet_scene_spec import (
    PyBulletSceneSpec,
)
from multitask_personalization.rom.models import LearnedROMModel


def test_learned_rom_model():
    """Tests for rom/models.py."""
    seed = 123

    rng = np.random.default_rng(seed)

    scene_spec = PyBulletSceneSpec()
    learned_rom_model = LearnedROMModel(scene_spec.human_spec)

    # Test LearnedROMModel
    parameters = learned_rom_model.get_trainable_parameters()
    assert isinstance(parameters, np.ndarray)
    assert parameters.shape == (4,)
    pts = learned_rom_model._reachable_points  # pylint: disable=protected-access
    n_prev_reachable_points = len(pts)
    # Add small gaussian noise to parameters.
    noise = rng.normal(0, 2e-1, len(parameters))
    learned_rom_model.set_trainable_parameters(parameters + noise)
    new_parameters = learned_rom_model.get_trainable_parameters()
    assert np.allclose(new_parameters, parameters + noise)
    pts = learned_rom_model._reachable_points  # pylint: disable=protected-access
    assert len(pts) != n_prev_reachable_points
