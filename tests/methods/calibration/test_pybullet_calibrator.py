"""Tests for pybullet_calibrator.py."""

import numpy as np

from multitask_personalization.envs.pybullet.pybullet_tasks import (
    PyBulletTask,
)
from multitask_personalization.methods.calibration.pybullet_calibrator import (
    PyBulletCalibrator,
)
from multitask_personalization.methods.interaction.random_interaction import (
    RandomInteractionMethod,
)


def test_grid_world_calibrator():
    """Tests for grid_world_calibrator.py."""

    task = PyBulletTask(
        intake_horizon=100,
        use_gui=False,
    )
    ip = task.intake_process
    im = RandomInteractionMethod(seed=123)
    ip.action_space.seed(123)
    im.reset(task.id, ip.action_space, ip.observation_space)
    calibrator = PyBulletCalibrator(task.scene_description)
    rng = np.random.default_rng(123)
    data = []
    for _ in range(100):
        action = im.get_action()
        obs = ip.sample_next_observation(action, rng)
        im.observe(obs)
        data.append((action, obs))
    parameters = calibrator.get_parameters(task.id, data)
    assert 0 < parameters < 1
