"""Tests for pybullet_env.py."""

from multitask_personalization.envs.pybullet.pybullet_env import PyBulletEnv
from multitask_personalization.envs.pybullet.pybullet_structs import PyBulletState
from multitask_personalization.envs.pybullet.pybullet_task_spec import PyBulletTaskSpec


def test_pybullet():
    """Tests for pybullet.py."""
    seed = 123

    task_spec = PyBulletTaskSpec()
    env = PyBulletEnv(task_spec, use_gui=False, seed=seed)
    env.action_space.seed(seed)
    obs, _ = env.reset()
    assert isinstance(obs, PyBulletState)

    for _ in range(10):
        act = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(act)
        assert isinstance(obs, PyBulletState)
        assert reward >= 0
        assert not terminated
        assert not truncated

    env.close()
