"""Tests for cooking_env.py."""

import os
from pathlib import Path

import numpy as np

from multitask_personalization.envs.cooking.cooking_env import (
    CookingEnv,
    CookingHiddenSpec,
    CookingSceneSpec,
    CookingState,
    WaitCookingAction,
)


def test_cooking_env():
    """Tests for cooking_env.py."""
    seed = 123

    scene_spec = CookingSceneSpec()
    hidden_spec = CookingHiddenSpec()
    env = CookingEnv(
        scene_spec,
        hidden_spec=hidden_spec,
        seed=seed,
    )

    # Uncomment to create video.
    # TODO comment out
    from gymnasium.wrappers import RecordVideo

    env = RecordVideo(env, "videos/test-cooking-env")

    env.action_space.seed(seed)
    obs, _ = env.reset()
    assert isinstance(obs, CookingState)

    for _ in range(10):
        act = WaitCookingAction()
        obs, reward, terminated, truncated, _ = env.step(act)
        assert np.isclose(reward, 0.0)
        assert isinstance(obs, CookingState)
        if terminated:
            break
        assert not truncated

    env.close()
