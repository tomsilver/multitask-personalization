"""Tests for pybullet_env.py."""

from multitask_personalization.envs.pybullet.pybullet_env import PyBulletEnv
from multitask_personalization.envs.pybullet.pybullet_structs import PyBulletState
from multitask_personalization.envs.pybullet.pybullet_task_spec import (
    HiddenTaskSpec,
    PyBulletTaskSpec,
)
from multitask_personalization.rom.models import GroundTruthROMModel


def test_pybullet():
    """Tests for pybullet.py."""
    seed = 123

    task_spec = PyBulletTaskSpec()
    preferred_books = ["book2"]
    rom_model = GroundTruthROMModel(task_spec.human_spec)
    hidden_spec = HiddenTaskSpec(book_preferences=preferred_books, rom_model=rom_model)
    env = PyBulletEnv(task_spec, hidden_spec=hidden_spec, use_gui=False, seed=seed)
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
