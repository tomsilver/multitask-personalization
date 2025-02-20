"""Tests for csp_approach.py."""

import numpy as np
import pytest

from multitask_personalization.csp_solvers import RandomWalkCSPSolver
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
from multitask_personalization.envs.tiny.tiny_env import (
    TinyEnv,
    TinyHiddenSpec,
    TinySceneSpec,
)
from multitask_personalization.methods.csp_approach import (
    CSPApproach,
)


@pytest.mark.parametrize(
    ["explore_method", "disable_learning"],
    [
        ("max-entropy", False),
        ("nothing-personal", False),
        ("exploit-only", False),
        ("epsilon-greedy", False),
        ("nothing-personal", True),
    ],
)
def test_csp_approach(explore_method, disable_learning):
    """Tests for csp_approach.py."""
    seed = 123
    scene_spec = TinySceneSpec()
    hidden_spec = TinyHiddenSpec(1.0, 0.5)
    solver = RandomWalkCSPSolver(seed, show_progress_bar=False)
    env = TinyEnv(scene_spec, hidden_spec=hidden_spec, seed=seed)
    approach = CSPApproach(
        scene_spec,
        env.action_space,
        solver,
        seed=seed,
        explore_method=explore_method,
        disable_learning=disable_learning,
    )
    approach.train()
    env.action_space.seed(seed)

    for _ in range(10):
        obs, info = env.reset()
        approach.reset(obs, info)
        for _ in range(100):
            act = approach.step()
            obs, reward, terminated, truncated, info = env.step(act)
            assert np.isclose(reward, 0.0)
            approach.update(obs, reward, terminated, info)
            assert not truncated
            if terminated:
                break

    env.close()


def test_cooking_csp_approach():
    """Tests CSP approach in cooking environment."""
    seed = 123

    universal_meal_specs = [
        MealSpec(
            "seasoning",
            [
                IngredientSpec("salt", temperature=(2.4, 3.6), quantity=(0.1, 2.0)),
                IngredientSpec("pepper", temperature=(2.4, 3.6), quantity=(0.1, 2.0)),
            ],
        )
    ]

    ground_truth_meal_specs = [
        MealSpec(
            "seasoning",
            [
                IngredientSpec("salt", temperature=(2.5, 3.5), quantity=(0.9, 1.1)),
                IngredientSpec("pepper", temperature=(2.5, 3.5), quantity=(0.9, 1.1)),
            ],
        )
    ]

    scene_spec = CookingSceneSpec(
        universal_meal_specs=universal_meal_specs,
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

    meal_model = MealSpecMealPreferenceModel(
        ground_truth_meal_specs, scene_spec.preference_shift_spec
    )
    hidden_spec = CookingHiddenSpec(meal_model)

    env = CookingEnv(
        scene_spec,
        hidden_spec=hidden_spec,
        seed=seed,
    )

    solver = RandomWalkCSPSolver(seed, show_progress_bar=False)
    approach = CSPApproach(
        scene_spec,
        env.action_space,
        solver,
        seed=seed,
        explore_method="max-entropy",
    )
    approach.train()
    env.action_space.seed(seed)

    obs, info = env.reset()
    approach.reset(obs, info)
    for _ in range(250):
        act = approach.step()
        obs, reward, terminated, truncated, info = env.step(act)
        assert np.isclose(reward, 0.0)
        approach.update(obs, reward, terminated, info)
        assert not truncated

    # Uncomment for debugging.
    # meal_model = approach._csp_generator._meal_model
    # for meal_spec in ground_truth_meal_specs:
    #     print(f"\n{meal_spec.name}")
    #     for ing_spec in meal_spec.ingredients:
    #         tf = meal_model._temperature_models[meal_spec.name][ing_spec.name]
    #         qf = meal_model._quantity_models[meal_spec.name][ing_spec.name]
    #         print(" ", ing_spec.name)
    #         print("   ", "temperature:")
    #         print("     ", "ground truth:", ing_spec.temperature)
    #         print("     ", "learned xs:", tf.x1, tf.x2, tf.x3, tf.x4)
    #         print("     ", "num data:", len(tf._incremental_X))
    #         print("     ", "num positive:", sum(tf._incremental_Y))
    #         print("   ", "quantity:")
    #         print("     ", "ground truth:", ing_spec.quantity)
    #         print("     ", "learned xs:", qf.x1, qf.x2, qf.x3, qf.x4)
    #         print("     ", "num data:", len(qf._incremental_X))
    #         print("     ", "num positive:", sum(qf._incremental_Y))

    env.close()
