"""Tests for cooking_env.py."""

import numpy as np

from multitask_personalization.envs.cooking.cooking_env import (
    AddIngredientCookingAction,
    CompleteCookingAction,
    CookingEnv,
    CookingHiddenSpec,
    CookingSceneSpec,
    CookingState,
    MovePotCookingAction,
    ServeMealCookingAction,
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
    # from gymnasium.wrappers import RecordVideo

    # env = RecordVideo(env, "videos/test-cooking-env")

    env.action_space.seed(seed)
    obs, _ = env.reset()
    assert isinstance(obs, CookingState)
    assert obs.meal_temperature is None

    # Put the pot on the stove.
    act = MovePotCookingAction(pot_id=0, new_pot_position=(3.0, 6.0))
    obs, reward, terminated, truncated, _ = env.step(act)
    assert np.isclose(reward, 0.0)
    assert not (terminated or truncated)
    assert isinstance(obs, CookingState)
    assert obs.pots[0].position == (3.0, 6.0)

    # Put another pot on the stove
    act = MovePotCookingAction(pot_id=1, new_pot_position=(3.0, 4.0))
    obs, reward, terminated, truncated, _ = env.step(act)
    assert np.isclose(reward, 0.0)
    assert not (terminated or truncated)
    assert isinstance(obs, CookingState)
    assert obs.pots[1].position == (3.0, 4.0)

    # Put another pot on the stove at an occupied location
    act = MovePotCookingAction(pot_id=2, new_pot_position=(3.0, 5.6))
    try:
        obs, reward, terminated, truncated, _ = env.step(act)
        assert False
    except ValueError as e:
        # Check that the error message is correct
        assert str(e) == "Pot position is already occupied."

    # Put another pot on the stove at an invalid location
    act = MovePotCookingAction(pot_id=2, new_pot_position=(0.0, 12.0))
    try:
        obs, reward, terminated, truncated, _ = env.step(act)
        assert False
    except ValueError as e:
        assert str(e) == "Pot position is outside the stove top."

    # Put the ingredient in the pot.
    act = AddIngredientCookingAction(pot_id=0, ingredient_id=0, ingredient_quantity=0.3)
    obs, reward, terminated, truncated, _ = env.step(act)
    assert np.isclose(reward, 0.0)
    assert not (terminated or truncated)
    assert isinstance(obs, CookingState)
    assert obs.pots[0].ingredient_in_pot_id == 0
    assert (
        obs.pots[0].ingredient_in_pot_temperature
        == scene_spec.ingredients[0].initial_temperature
    )
    assert np.isclose(obs.pots[0].ingredient_quantity_in_pot, 0.3)

    # Add to pot not on stove
    act = AddIngredientCookingAction(pot_id=2, ingredient_id=1, ingredient_quantity=0.1)
    try:
        obs, reward, terminated, truncated, _ = env.step(act)
        assert False
    except ValueError as e:
        assert str(e) == "Cannot add ingredient to a pot that is not on the stove."

    # Add to non-empty pot
    act = AddIngredientCookingAction(pot_id=0, ingredient_id=1, ingredient_quantity=0.1)
    try:
        obs, reward, terminated, truncated, _ = env.step(act)
        assert False
    except ValueError as e:
        assert str(e) == "Can only add ingredients to empty pots."

    # Add more than available
    act = AddIngredientCookingAction(pot_id=1, ingredient_id=0, ingredient_quantity=5.0)
    try:
        obs, reward, terminated, truncated, _ = env.step(act)
        assert False
    except ValueError as e:
        assert str(e) == "Not enough unused ingredient to add."

    # Add more than the pot can hold
    act = AddIngredientCookingAction(
        pot_id=1, ingredient_id=3, ingredient_quantity=18.0
    )
    try:
        obs, reward, terminated, truncated, _ = env.step(act)
        assert False
    except ValueError as e:
        assert str(e) == "Cannot exceed the pot's capacity."

    # Add the ingredient to the pot.
    act = AddIngredientCookingAction(pot_id=1, ingredient_id=3, ingredient_quantity=0.6)
    obs, reward, terminated, truncated, _ = env.step(act)
    assert np.isclose(reward, 0.0)
    assert not (terminated or truncated)
    assert isinstance(obs, CookingState)
    assert obs.pots[1].ingredient_in_pot_id == 3
    assert (
        obs.pots[1].ingredient_in_pot_temperature
        == scene_spec.ingredients[3].initial_temperature
    )
    assert np.isclose(obs.pots[1].ingredient_quantity_in_pot, 0.6)

    # Wait.
    act = WaitCookingAction()
    obs, reward, terminated, truncated, _ = env.step(act)
    assert np.isclose(reward, 0.0)
    assert not (terminated or truncated)
    assert isinstance(obs, CookingState)
    assert (
        obs.pots[0].ingredient_in_pot_temperature
        == scene_spec.ingredients[0].initial_temperature
        + scene_spec.ingredients[0].temperature_increase_rate
        * 6  # number of actions called
    )
    assert obs.pots[0].ingredient_quantity_in_pot == 0.3
    assert (
        obs.pots[1].ingredient_in_pot_temperature
        == scene_spec.ingredients[3].initial_temperature
        + scene_spec.ingredients[3].temperature_increase_rate
        * 1  # number of actions called
    )
    assert np.isclose(obs.pots[1].ingredient_quantity_in_pot, 0.6)

    # Keep cooking until cooked
    for _ in range(14):
        act = WaitCookingAction()
        obs, reward, terminated, truncated, _ = env.step(act)

    # Check that pot 0 is done
    assert obs.pots[0].ingredient_in_pot_id == 0
    assert np.isclose(obs.pots[0].ingredient_quantity_in_pot, 0.3)
    assert np.isclose(obs.pots[0].ingredient_in_pot_temperature, 10.0)
    assert obs.pots[0].ingredient_done

    # Check that pot 1 is not done
    assert obs.pots[1].ingredient_in_pot_id == 3
    assert np.isclose(obs.pots[1].ingredient_quantity_in_pot, 0.6)
    assert np.isclose(obs.pots[1].ingredient_in_pot_temperature, 1.5)
    assert not obs.pots[1].ingredient_done

    # Check that CompleteCooking isn't available yet
    act = CompleteCookingAction()
    try:
        obs, reward, terminated, truncated, _ = env.step(act)
        assert False
    except ValueError as e:
        assert str(e) == "Cannot complete cooking with uncooked ingredients."

    # Keep cooking until cooked
    for _ in range(84):
        act = WaitCookingAction()
        obs, reward, terminated, truncated, _ = env.step(act)

    # Check that pot 1 is done
    assert np.isclose(obs.pots[1].ingredient_in_pot_temperature, 10.0)
    assert obs.pots[1].ingredient_done

    assert obs.meal_temperature is None

    # Complete cooking
    act = CompleteCookingAction()
    obs, reward, terminated, truncated, _ = env.step(act)

    # Check that the meal temperature is correctly intialized
    assert np.isclose(obs.meal_temperature, 10.0)

    # Wait for meal to cool
    for _ in range(20):
        act = WaitCookingAction()
        obs, reward, terminated, truncated, _ = env.step(act)
    assert np.isclose(
        obs.meal_temperature,
        scene_spec.cooked_temperature - scene_spec.meal_cooling_rate * 20,
    )

    # Serve.
    act = ServeMealCookingAction()
    obs, reward, terminated, truncated, info = env.step(act)
    assert np.isclose(reward, 0.0)
    assert not (terminated or truncated)
    assert isinstance(obs, CookingState)
    assert obs.pots[0].position is None
    assert obs.pots[0].ingredient_in_pot_id is None
    assert info["user_satisfaction"] != 0

    env.close()
