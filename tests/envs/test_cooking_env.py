"""Tests for cooking_env.py."""

import numpy as np

from multitask_personalization.envs.cooking.cooking_env import CookingEnv
from multitask_personalization.envs.cooking.cooking_hidden_spec import (
    CookingHiddenSpec,
    MealSpecMealPreferenceModel,
)
from multitask_personalization.envs.cooking.cooking_meals import (
    IngredientSpec,
    MealSpec,
)
from multitask_personalization.envs.cooking.cooking_scene_spec import (
    CookingIngredient,
    CookingPot,
    CookingSceneSpec,
)
from multitask_personalization.envs.cooking.cooking_structs import (
    AddIngredientCookingAction,
    CookingState,
    MovePotCookingAction,
    MultiCookingAction,
    ServeMealCookingAction,
    WaitCookingAction,
)


def test_cooking_env():
    """Tests for cooking_env.py."""
    seed = 123

    scene_spec = CookingSceneSpec()
    meal_specs = [
        MealSpec(
            "seasoning",
            [
                IngredientSpec("salt", temperature=(2.0, 4.0), quantity=(0.9, 1.1)),
                IngredientSpec("pepper", temperature=(2.0, 4.0), quantity=(0.9, 1.1)),
            ],
        )
    ]
    meal_model = MealSpecMealPreferenceModel(meal_specs)
    hidden_spec = CookingHiddenSpec(meal_model)
    env = CookingEnv(
        scene_spec,
        hidden_spec=hidden_spec,
        seed=seed,
    )

    env.action_space.seed(seed)
    obs, _ = env.reset()
    assert isinstance(obs, CookingState)

    # Put the pot on the stove.
    act = MovePotCookingAction(pot_id=0, new_pot_position=(3.0, 6.0))
    obs, reward, terminated, truncated, _ = env.step(act)
    assert np.isclose(reward, 0.0)
    assert not (terminated or truncated)
    assert isinstance(obs, CookingState)
    assert obs.pots[0].position == (3.0, 6.0)

    # Put another pot on the stove.
    act = MovePotCookingAction(pot_id=1, new_pot_position=(3.0, 4.0))
    obs, reward, terminated, truncated, _ = env.step(act)
    assert np.isclose(reward, 0.0)
    assert not (terminated or truncated)
    assert isinstance(obs, CookingState)
    assert obs.pots[1].position == (3.0, 4.0)

    # Put another pot on the stove at an occupied location.
    act = MovePotCookingAction(pot_id=2, new_pot_position=(3.0, 5.6))
    obs, reward, terminated, truncated, _ = env.step(act)
    assert obs.pots[2].position is None  # action failed

    # Put another pot on the stove at an invalid location.
    act = MovePotCookingAction(pot_id=2, new_pot_position=(0.0, 12.0))
    obs, reward, terminated, truncated, _ = env.step(act)
    assert obs.pots[2].position is None  # action failed

    # Put an ingredient in a pot.
    act = AddIngredientCookingAction(
        pot_id=0, ingredient="salt", ingredient_quantity=0.3
    )
    obs, reward, terminated, truncated, _ = env.step(act)
    assert np.isclose(reward, 0.0)
    assert not (terminated or truncated)
    assert isinstance(obs, CookingState)
    assert obs.pots[0].ingredient_in_pot == "salt"
    assert obs.pots[0].ingredient_in_pot_temperature == 0.0
    assert np.isclose(obs.pots[0].ingredient_quantity_in_pot, 0.3)

    # Add to non-empty pot.
    act = AddIngredientCookingAction(
        pot_id=0, ingredient="pepper", ingredient_quantity=0.1
    )
    initial_pepper_state = obs.ingredients["pepper"]
    obs, reward, terminated, truncated, _ = env.step(act)
    assert obs.ingredients["pepper"] == initial_pepper_state  # action failed
    assert obs.pots[0].ingredient_in_pot == "salt"

    # Add more than available.
    act = AddIngredientCookingAction(
        pot_id=1, ingredient="pepper", ingredient_quantity=5.0
    )
    obs, reward, terminated, truncated, _ = env.step(act)
    assert obs.ingredients["pepper"] == initial_pepper_state  # action failed

    # Test increasing temperature.
    act = WaitCookingAction()
    current_salt_temperature = obs.pots[0].ingredient_in_pot_temperature
    obs, reward, terminated, truncated, _ = env.step(act)
    next_salt_temperature = obs.pots[0].ingredient_in_pot_temperature
    delta = scene_spec.get_ingredient("salt").heat_rate
    assert np.isclose(next_salt_temperature - current_salt_temperature, delta)

    # Test decreasing temperature (which requires removing the pot first).
    act = MovePotCookingAction(pot_id=0, new_pot_position=None)
    obs, reward, terminated, truncated, _ = env.step(act)
    assert obs.pots[0].position is None
    current_salt_temperature = obs.pots[0].ingredient_in_pot_temperature
    act = WaitCookingAction()
    obs, reward, terminated, truncated, _ = env.step(act)
    next_salt_temperature = obs.pots[0].ingredient_in_pot_temperature
    delta = scene_spec.get_ingredient("salt").cool_rate
    assert np.isclose(current_salt_temperature - next_salt_temperature, delta)

    # Test serving (user will be unhappy).
    act = ServeMealCookingAction("seasoning")
    obs, reward, terminated, truncated, info = env.step(act)
    assert np.isclose(reward, 0.0)
    assert terminated
    assert isinstance(obs, CookingState)
    assert info["user_satisfaction"] == -1

    env.close()


def test_cooking_env_full_meal():
    """Test creating and serving a successful meal in the cooking env."""
    seed = 123

    scene_spec = CookingSceneSpec(
        universal_meal_specs=[
            MealSpec(
                "seasoning",
                [
                    IngredientSpec("salt", temperature=(2.5, 3.5), quantity=(0.9, 1.1)),
                    IngredientSpec(
                        "pepper", temperature=(2.5, 3.5), quantity=(0.9, 1.1)
                    ),
                ],
            )
        ],
        pots=[
            CookingPot(radius=0.5, position=None),
            CookingPot(radius=1.0, position=None),
        ],
        ingredients=[
            CookingIngredient(
                name="salt",
                color=(0.9, 0.9, 0.9),
                respawn_quantity_bounds=(1.0, 1.1),
                heat_rate=1.0,
                cool_rate=1.0,
            ),
            CookingIngredient(
                name="pepper",
                color=(0.0, 0.0, 0.0),
                respawn_quantity_bounds=(1.0, 1.1),
                heat_rate=1.0,
                cool_rate=1.0,
            ),
        ],
    )
    meal_model = MealSpecMealPreferenceModel(scene_spec.universal_meal_specs)
    hidden_spec = CookingHiddenSpec(meal_model)

    env = CookingEnv(
        scene_spec,
        hidden_spec=hidden_spec,
        seed=seed,
    )

    # Uncomment to create video.
    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/test-cooking-env")

    plan = [
        MovePotCookingAction(0, (3.0, 6.0)),
        MovePotCookingAction(1, (8.0, 3.0)),
        MultiCookingAction(
            [
                AddIngredientCookingAction(
                    pot_id=0,
                    ingredient="salt",
                    ingredient_quantity=1.0,
                ),
                AddIngredientCookingAction(
                    pot_id=1,
                    ingredient="pepper",
                    ingredient_quantity=1.0,
                ),
            ]
        ),
        WaitCookingAction(),  # 0 -> 1
        WaitCookingAction(),  # 1 -> 2
        WaitCookingAction(),  # 2 -> 3
        ServeMealCookingAction("seasoning"),
    ]

    # Repeat twice.
    for _ in range(2):
        env.reset()
        for act in plan:
            _, _, terminated, _, info = env.step(act)
        assert terminated
        assert info["user_satisfaction"] == 1.0

    env.close()
