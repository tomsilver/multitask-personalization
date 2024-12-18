"""CSP generation for the cooking environment."""

from pathlib import Path
from typing import Any, get_args
from dataclasses import dataclass

import numpy as np

from multitask_personalization.csp_generation import CSPGenerator
from multitask_personalization.envs.cooking.cooking_structs import (
    CookingAction,
    CookingState,
)
from multitask_personalization.envs.cooking.cooking_scene_spec import CookingSceneSpec
from multitask_personalization.envs.cooking.cooking_hidden_spec import CookingHiddenSpec
from tomsutils.spaces import FunctionalSpace

from multitask_personalization.structs import (
    CSP,
    CSPConstraint,
    CSPCost,
    CSPPolicy,
    CSPSampler,
    CSPVariable,
    FunctionalCSPConstraint,
    FunctionalCSPSampler,
)


@dataclass(frozen=True)
class _IngredientCSPState:

    name: str
    is_used: bool
    pot_id: int
    start_time: int
    pos: tuple[float, float]



class CookingCSPGenerator(CSPGenerator[CookingState, CookingAction]):
    """Generates CSPs for the cooking environment."""

    def __init__(
        self,
        scene_spec: CookingSceneSpec,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._scene_spec = scene_spec

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
        
        variable_space = FunctionalSpace(
            contains_fn=lambda x: isinstance(x, get_args(_IngredientCSPState)),
        )
        
        # One per ingredient with: is_used, quantity, pot_id, start_time.
        ingredients = sorted(obs.ingredients)
        variables = [CSPVariable(i, variable_space) for i in ingredients]

        # Initialization.
        initialization = {
            v: _IngredientCSPState(v.name, False, 0, 0, (0.0, 0.0))
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
            *ingredients: _IngredientCSPState,
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
        def _pots_nonoverlapping(ingredient1: _IngredientCSPState, ingredient2: _IngredientCSPState) -> bool:
            pot1 = ingredient1.pot_id
            pot2 = ingredient2.pot_id
            r1 = self._scene_spec.pots[pot1].radius
            r2 = self._scene_spec.pots[pot2].radius
            dist = np.linalg.norm(
                    np.array(ingredient1.pos) - np.array(ingredient2.pos)
                )
            return dist >= r1 + r2

        for i, ingredient1 in enumerate(variables[:-1]):
            for ingredient2 in variables[i+1:]:
                constraint = FunctionalCSPConstraint(
                    f"{ingredient1.name}-{ingredient2.name}-nonoverlapping",
                    [ingredient1, ingredient2],
                    _pots_nonoverlapping
                )
                constraints.append(constraint)

        # Ingredients cannot be in the same pot.
        def _ingredients_different_pots(ingredient1: _IngredientCSPState, ingredient2: _IngredientCSPState) -> bool:
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
        def _ingredient_quantity_exists(ingredient: _IngredientCSPState) -> bool:
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
        return _CookingCSPPolicy(csp, seed=self._seed)

    def observe_transition(
        self,
        obs: CookingState,
        act: CookingAction,
        next_obs: CookingState,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        pass


class _CookingCSPPolicy(CSPPolicy[CookingState, CookingAction]):
    
    def step(self, obs: CookingState) -> CookingAction:
        import ipdb; ipdb.set_trace()

    def check_termination(self, obs: CookingState) -> bool:
        import ipdb; ipdb.set_trace()
