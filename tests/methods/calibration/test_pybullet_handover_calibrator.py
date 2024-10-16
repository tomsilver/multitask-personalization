"""Tests for pybullet_handover_calibrator.py."""

import numpy as np

from multitask_personalization.envs.pybullet.pybullet_handover import (
    PyBulletHandoverTask,
)
from multitask_personalization.methods.calibration.pybullet_handover_calibrator import (
    PyBulletHandoverCalibrator,
)
from multitask_personalization.methods.interaction.random_interaction import (
    RandomInteractionMethod,
)


def test_grid_world_calibrator():
    """Tests for grid_world_calibrator.py."""

    task = PyBulletHandoverTask(
        intake_horizon=100,
        use_gui=False,
    )
    ip = task.intake_process
    im = RandomInteractionMethod(seed=123)
    ip.action_space.seed(123)
    im.reset(task.id, ip.action_space, ip.observation_space)
    calibrator = PyBulletHandoverCalibrator(task.scene_description)
    rng = np.random.default_rng(123)
    data = []
    for _ in range(100):
        action = im.get_action()
        obs = ip.sample_next_observation(action, rng)
        im.observe(obs)
        data.append((action, obs))
    parameters = calibrator.get_parameters(task.id, data)
    assert 0 < parameters < 1
