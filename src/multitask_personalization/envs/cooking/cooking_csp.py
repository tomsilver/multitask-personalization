"""CSP generation for the cooking environment."""

import abc
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.spaces import Box, Discrete, Tuple
from numpy.typing import NDArray
from tomsutils.spaces import EnumSpace
from tomsutils.utils import create_rng_from_rng

from multitask_personalization.csp_generation import CSPGenerator
from multitask_personalization.envs.cooking.cooking_structs import (
    CookingAction,
    CookingState,
)
from multitask_personalization.structs import (
    CSP,
    CSPConstraint,
    CSPCost,
    CSPPolicy,
    CSPSampler,
    CSPVariable,
    FunctionalCSPConstraint,
    FunctionalCSPSampler,
    LogProbCSPConstraint,
)


class CookingCSPGenerator(CSPGenerator[CookingState, CookingAction]):
    """Generates CSPs for the cooking environment."""

    def save(self, model_dir: Path) -> None:
        # Nothing is learned yet.
        pass

    def load(self, model_dir: Path) -> None:
        # Nothing is learned yet.
        pass

    def _generate_variables(
        self,
        obs: CookingState,
    ) -> tuple[list[CSPVariable], dict[CSPVariable, Any]]:
        
        # One per ingredient with: is_used, quantity, pot_id, start_time.
        ingredients = sorted(obs.ingredients)
        variables = [
            CSPVariable(ingredient, Box(0, np.inf, (4,), dtype=np.float_))
            for ingredient in ingredients
        ]

        # Initialization.
        initialization = {
            v: np.zeros(4, dtype=np.float_)
            for v in variables
        }

        return variables, initialization

    def _generate_personal_constraints(
        self,
        obs: CookingState,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:
        
        # Final ingredients must comprise some happy meal.
        def _is_happy_meal(
            *ingredients: NDArray,
        ) -> bool:
            import ipdb; ipdb.set_trace()

        happy_meal_constraint = FunctionalCSPConstraint(
            "happy_meal_constraint",
            variables,
            _is_happy_meal,
        )

        return [happy_meal_constraint]


    def _generate_nonpersonal_constraints(
        self,
        obs: CookingState,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:
        
        constraints: list[CSPConstraint] = []
        
        # Pots cannot overlap on the stove.
        def _pots_nonoverlapping(ingredient1: NDArray, ingredient2: NDArray) -> bool:
            import ipdb; ipdb.set_trace()

        for i, ingredient1 in enumerate(variables[:-1]):
            for ingredient2 in variables[i+1:]:
                constraint = FunctionalCSPConstraint(
                    f"{ingredient1.name}-{ingredient2.name}-nonoverlapping",
                    [ingredient1, ingredient2],
                    _pots_nonoverlapping
                )
                constraints.append(constraint)

        # Ingredients cannot be in the same pot.
        def _ingredients_different_pots(ingredient1: NDArray, ingredient2: NDArray) -> bool:
            import ipdb; ipdb.set_trace()

        for i, ingredient1 in enumerate(variables[:-1]):
            for ingredient2 in variables[i+1:]:
                constraint = FunctionalCSPConstraint(
                    f"{ingredient1.name}-{ingredient2.name}-different-pots",
                    [ingredient1, ingredient2],
                    _ingredients_different_pots
                )
                constraints.append(constraint)

        # Ingredient quantity used must be not more than total available.
        def _ingredient_quantity_exists(ingredient: NDArray) -> bool:
            import ipdb; ipdb.set_trace()

        for ingredient in variables:
            constraint = FunctionalCSPConstraint(
                f"{ingredient.name}-quantity-exists",
                [ingredient],
                _ingredient_quantity_exists
            )
            constraints.append(constraint)

        return constraints
        

    def _generate_exploit_cost(
        self,
        obs: CookingState,
        variables: list[CSPVariable],
    ) -> CSPCost | None:
        return None

    def _generate_samplers(
        self,
        obs: CookingState,
        csp: CSP,
    ) -> list[CSPSampler]:

        # Sample ingredients by internally sampling a happy meal.
        def _sample_ingredients(
            _: dict[CSPVariable, Any], rng: np.random.Generator
        ) -> dict[CSPVariable, Any]:
            import ipdb; ipdb.set_trace()

        ingredient_sampler = FunctionalCSPSampler(_sample_ingredients, csp, set(csp.variables))

        # Sample positions for all currently used pots.
        def _sample_pot_positions(
            _: dict[CSPVariable, Any], rng: np.random.Generator
        ) -> dict[CSPVariable, Any]:
            import ipdb; ipdb.set_trace()

        pot_position_sampler = FunctionalCSPSampler(_sample_pot_positions, csp, set(csp.variables))

        return [ingredient_sampler, pot_position_sampler]

    def _generate_policy(
        self,
        obs: CookingState,
        csp: CSP,
    ) -> CSPPolicy:
        import ipdb

        ipdb.set_trace()

    def observe_transition(
        self,
        obs: CookingState,
        act: CookingAction,
        next_obs: CookingState,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        pass
