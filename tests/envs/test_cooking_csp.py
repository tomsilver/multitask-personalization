"""Tests for cooking_csp.py."""

import numpy as np

from multitask_personalization.csp_solvers import RandomWalkCSPSolver
from multitask_personalization.envs.cooking.cooking_csp import (
    CookingCSPGenerator,
)
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
    CookingState,
)


def test_cooking_csp():
    """Tests for cooking_csp.py."""

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

    env.action_space.seed(seed)
    obs, _ = env.reset()
    assert isinstance(obs, CookingState)

    # Create the CSP.
    csp_generator = CookingCSPGenerator(scene_spec, meal_model, seed=seed)
    csp, samplers, policy, initialization = csp_generator.generate(obs)

    # Solve the CSP.
    solver = RandomWalkCSPSolver(seed, show_progress_bar=False)
    sol = solver.solve(
        csp,
        initialization,
        samplers,
    )
    assert sol is not None
    policy.reset(sol)

    # Run the policy.
    for _ in range(1000):
        act = policy.step(obs)
        obs, reward, env_terminated, truncated, info = env.step(act)
        assert isinstance(obs, CookingState)
        assert np.isclose(reward, 0.0)
        if env_terminated:
            assert info["user_satisfaction"] == 1.0
            break
        assert not truncated
    else:
        assert False, "Policy failed."

    env.close()
