"""Tests for random_actions_approach.py."""

from multitask_personalization.envs.pybullet.pybullet_env import PyBulletEnv
from multitask_personalization.envs.pybullet.pybullet_structs import PyBulletState
from multitask_personalization.envs.pybullet.pybullet_task_spec import PyBulletTaskSpec
from multitask_personalization.methods.random_actions_approach import (
    RandomActionsApproach,
)


def test_random_actions_approach():
    """Tests for random_actions_approach.py."""
    seed = 123

    task_spec = PyBulletTaskSpec()
    env = PyBulletEnv(task_spec, use_gui=False, seed=seed)
    approach = RandomActionsApproach(env.action_space, seed=seed)
    approach.eval()
    env.action_space.seed(seed)
    obs, _ = env.reset()
    approach.reset(obs)
    assert isinstance(obs, PyBulletState)

    for _ in range(10):
        act = approach.step()
        obs, reward, terminated, truncated, _ = env.step(act)
        approach.update(obs, reward, terminated)
        assert isinstance(obs, PyBulletState)
        assert reward >= 0
        assert not terminated
        assert not truncated

    env.close()
