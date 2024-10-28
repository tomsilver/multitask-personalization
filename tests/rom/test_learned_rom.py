"""Tests for rom/models.py."""

import numpy as np

from multitask_personalization.envs.pybullet.pybullet_task_spec import (
    PyBulletTaskSpec,
)
from multitask_personalization.rom.models import LearnedROMModel


def test_learned_rom_model():
    """Tests for rom/models.py."""
    seed = 123

    rng = np.random.default_rng(seed)

    task_spec = PyBulletTaskSpec()
    learned_rom_model = LearnedROMModel(task_spec.human_spec, seed=seed)

    # Test LearnedROMModel
    assert (
        len(learned_rom_model.get_rom_model_context_parameters())
        == learned_rom_model.get_parameter_size()
    )
    # Test update_parameters is updating the parameters and reachable_points
    parameters = learned_rom_model.get_rom_model_context_parameters()
    # add small gaussian noise to parameters
    n_prev_reachable_points = len(learned_rom_model.get_reachable_points())
    noise = rng.normal(0, 2e-1, len(parameters))
    learned_rom_model.update_parameters(parameters + noise)
    assert np.allclose(
        learned_rom_model.get_rom_model_context_parameters(), parameters + noise
    )
    assert len(learned_rom_model.get_reachable_points()) != n_prev_reachable_points
