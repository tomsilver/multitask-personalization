"""Tests for pybullet_env.py."""

import os

from multitask_personalization.envs.pybullet.pybullet_env import PyBulletEnv
from multitask_personalization.envs.pybullet.pybullet_structs import PyBulletState
from multitask_personalization.envs.pybullet.pybullet_task_spec import (
    HiddenTaskSpec,
    PyBulletTaskSpec,
)
from multitask_personalization.rom.models import SphericalROMModel


def test_pybullet():
    """Tests for pybullet.py."""
    os.environ["OPENAI_API_KEY"] = "NOT A REAL KEY"  # will not be used
    seed = 123

    task_spec = PyBulletTaskSpec()
    book_preferences = (
        "I enjoy fiction, especially science fiction, but I hate nonfiction"
    )
    rom_model = SphericalROMModel(task_spec.human_spec)
    hidden_spec = HiddenTaskSpec(book_preferences=book_preferences, rom_model=rom_model)
    env = PyBulletEnv(
        task_spec,
        hidden_spec=hidden_spec,
        use_gui=False,
        seed=seed,
        llm_use_cache_only=True,
    )
    env.action_space.seed(seed)
    obs, _ = env.reset()
    assert isinstance(obs, PyBulletState)

    for _ in range(10):
        act = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(act)
        assert isinstance(obs, PyBulletState)
        if terminated:
            assert reward in {-1, 1}
            break
        assert reward == 0
        assert not truncated

    env.close()
