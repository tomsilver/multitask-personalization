"""Tests for pybullet_env.py."""

import os
from pathlib import Path

import numpy as np
import pytest
from tomsutils.llm import OpenAILLM

from multitask_personalization.envs.pybullet.pybullet_env import PyBulletEnv
from multitask_personalization.envs.pybullet.pybullet_scene_spec import (
    HiddenSceneSpec,
    PyBulletSceneSpec,
)
from multitask_personalization.envs.pybullet.pybullet_structs import PyBulletState
from multitask_personalization.envs.pybullet.pybullet_utils import PyBulletCannedLLM
from multitask_personalization.rom.models import SphericalROMModel

_LLM_CACHE_DIR = Path(__file__).parents[1] / "unit_test_llm_cache"
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "NOT A REAL KEY"  # will not be used


@pytest.mark.parametrize(
    "llm",
    [
        # OpenAILLM(
        #     model_name="gpt-4o-mini",
        #     cache_dir=_LLM_CACHE_DIR,
        #     max_tokens=700,
        #     use_cache_only=True,
        # ),
        PyBulletCannedLLM(_LLM_CACHE_DIR),
    ],
)
def test_pybullet(llm):
    """Tests for pybullet.py."""
    seed = 123

    default_scene_spec = PyBulletSceneSpec()
    scene_spec = PyBulletSceneSpec(
        book_half_extents=default_scene_spec.book_half_extents[:3],
        book_poses=default_scene_spec.book_poses[:3],
    )
    book_preferences = "I like pretty much anything!"
    rom_model = SphericalROMModel(scene_spec.human_spec)
    surfaces_robot_can_clean = [
        ("table", -1),
        ("shelf", 0),
        ("shelf", 1),
        ("shelf", 2),
    ]
    hidden_spec = HiddenSceneSpec(
        book_preferences=book_preferences,
        rom_model=rom_model,
        surfaces_robot_can_clean=surfaces_robot_can_clean,
    )
    env = PyBulletEnv(
        scene_spec,
        llm,
        hidden_spec=hidden_spec,
        use_gui=True,
        seed=seed,
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
