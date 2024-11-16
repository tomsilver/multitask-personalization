"""Tests for pybullet_env.py."""

import os
from pathlib import Path

import numpy as np

from multitask_personalization.envs.pybullet.pybullet_env import PyBulletEnv
from multitask_personalization.envs.pybullet.pybullet_scene_spec import (
    HiddenSceneSpec,
    PyBulletSceneSpec,
)
from multitask_personalization.envs.pybullet.pybullet_structs import PyBulletState
from multitask_personalization.rom.models import SphericalROMModel


def test_pybullet():
    """Tests for pybullet.py."""
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "NOT A REAL KEY"  # will not be used
    seed = 123

    default_scene_spec = PyBulletSceneSpec()
    scene_spec = PyBulletSceneSpec(
        book_half_extents=default_scene_spec.book_half_extents[:3],
        book_poses=default_scene_spec.book_poses[:3],
        book_rgbas=default_scene_spec.book_rgbas[:3],
    )
    book_preferences = "I like pretty much anything!"
    rom_model = SphericalROMModel(scene_spec.human_spec)
    hidden_spec = HiddenSceneSpec(
        book_preferences=book_preferences, rom_model=rom_model
    )
    env = PyBulletEnv(
        scene_spec,
        hidden_spec=hidden_spec,
        use_gui=False,
        seed=seed,
        llm_cache_dir=Path(__file__).parents[1] / "unit_test_llm_cache",
        llm_use_cache_only=True,
    )
    env.action_space.seed(seed)
    obs, _ = env.reset()
    assert isinstance(obs, PyBulletState)

    for _ in range(10):
        act = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(act)
        assert np.isclose(reward, 0.0)
        assert isinstance(obs, PyBulletState)
        if terminated:
            break
        assert not truncated

    env.close()
