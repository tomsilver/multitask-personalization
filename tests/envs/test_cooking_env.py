"""Tests for cooking_env.py."""

import os
from pathlib import Path

import numpy as np

from multitask_personalization.envs.cooking.cooking_env import (
    AddIngredientCookingAction,
    CookingEnv,
    CookingHiddenSpec,
    CookingSceneSpec,
    CookingState,
    MovePotCookingAction,
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

    # Put the pot on the stove.
    act = MovePotCookingAction(new_pot_position=(0.5, 0.5))
    obs, reward, terminated, truncated, _ = env.step(act)
    assert np.isclose(reward, 0.0)
    assert isinstance(obs, CookingState)
    assert obs.pot_position == (0.5, 0.5)

    # Put the ingredient in the pot.
    act = AddIngredientCookingAction("salt", 0.3)
    obs, reward, terminated, truncated, _ = env.step(act)
    assert np.isclose(reward, 0.0)
    assert isinstance(obs, CookingState)
    assert obs.ingredient_in_pot == "salt"
    assert (
        obs.ingredient_in_pot_temperature == scene_spec.initial_ingredient_temperature
    )
    assert obs.ingredient_quantity_in_pot == 0.3

    # Wait.
    act = WaitCookingAction()
    obs, reward, terminated, truncated, _ = env.step(act)
    assert np.isclose(reward, 0.0)
    assert isinstance(obs, CookingState)
    assert obs.ingredient_in_pot == "salt"
    assert (
        obs.ingredient_in_pot_temperature
        == scene_spec.initial_ingredient_temperature
        + scene_spec.ingredient_temperature_increase_rate
    )
    assert obs.ingredient_quantity_in_pot == 0.3

    env.close()
