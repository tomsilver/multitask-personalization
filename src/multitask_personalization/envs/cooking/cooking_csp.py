"""CSP generation for the cooking environment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

import numpy as np
from tomsutils.spaces import FunctionalSpace
from gymnasium.spaces import Discrete
from multitask_personalization.utils import _NoChange, _NO_CHANGE


from multitask_personalization.csp_generation import CSPGenerator
from multitask_personalization.envs.cooking.cooking_hidden_spec import MealPreferenceModel
from multitask_personalization.envs.cooking.cooking_scene_spec import CookingSceneSpec
from multitask_personalization.envs.cooking.cooking_structs import (
    CookingAction,
    CookingState,
    Meal,
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
)


@dataclass(frozen=True)
class _IngredientCSPState:

    name: str
    is_used: bool
    quantity: float
    pot_id: int
    start_time: int
    pos: tuple[float, float]
    
    def calculate_final_temperature(self, scene_spec: CookingSceneSpec,
                                 total_time: int) -> float:
        """Calculate the final ingredient temperature."""
        ingredient_cooking_time = total_time - self.start_time
        if ingredient_cooking_time <= 0:
            return 0.0
        heat_rate = scene_spec.get_ingredient(self.name).heat_rate
        return ingredient_cooking_time * heat_rate
    
    def copy_with(self, is_used: bool | _NoChange = _NO_CHANGE,
                  quantity: float | _NoChange = _NO_CHANGE,
                  pot_id: int | _NoChange = _NO_CHANGE,
                  start_time: float | _NoChange = _NO_CHANGE,
                  pos: tuple[float, float] | _NoChange = _NO_CHANGE
                  ) -> _IngredientCSPState:
        """Create a new ingredient state."""
        return _IngredientCSPState(
            self.name,
            is_used=self.is_used if is_used is _NO_CHANGE else is_used,
            quantity=self.quantity if quantity is _NO_CHANGE else quantity,
            pot_id=self.pot_id if pot_id is _NO_CHANGE else pot_id,
            start_time=self.start_time if start_time is _NO_CHANGE else start_time,
            pos=self.pos if pos is _NO_CHANGE else pos,
        )


class CookingCSPGenerator(CSPGenerator[CookingState, CookingAction]):
    """Generates CSPs for the cooking environment."""

    def __init__(
        self,
        scene_spec: CookingSceneSpec,
        meal_model: MealPreferenceModel,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._scene_spec = scene_spec
        self._meal_model = meal_model

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
        ingredient_variables = [CSPVariable(i, variable_space) for i in ingredients]

        # One "global" variable for the total amount of cooking time.
        cooking_time = CSPVariable("total-cooking-time", Discrete(10000000))
        variables = ingredient_variables + [cooking_time]

        # Initialization.
        initialization = {
            v: _IngredientCSPState(v.name, False, 0, -1, 0, (-1, -1)) for v in ingredient_variables
        }
        initialization[cooking_time] = 0.0

        return variables, initialization

    def _generate_personal_constraints(
        self,
        obs: CookingState,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:

        ingredient_variables = variables[:-1]
        cooking_time_variable = variables[-1]

        # Final ingredients must comprise some meal that the user enjoys.
        def _user_enjoys_meal(
            total_cooking_time: float,
            *ingredients: _IngredientCSPState,
        ) -> bool:
            # Derive meal.
            meal_ingredients = {}
            for ing_state in ingredients:
                if not ing_state.is_used:
                    continue
                temp = ing_state.calculate_final_temperature(self._scene_spec, total_cooking_time)
                quant = ing_state.quantity
                meal_ingredients[ing_state.name] = (temp, quant)
            meal = Meal(meal_ingredients)
            # Check against model.
            return self._meal_model.check(meal)

        user_enjoys_meal_constraint = FunctionalCSPConstraint(
            "user_enjoys_meal_constraint",
            [cooking_time_variable] + ingredient_variables,
            _user_enjoys_meal,
        )

        return [user_enjoys_meal_constraint]

    def _generate_nonpersonal_constraints(
        self,
        obs: CookingState,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:

        constraints: list[CSPConstraint] = []
        ingredient_variables = variables[:-1]

        # Pots cannot overlap on the stove.
        def _pots_nonoverlapping(
            ingredient1: _IngredientCSPState, ingredient2: _IngredientCSPState
        ) -> bool:
            # Unused pots need not be checked.
            if not ingredient1.is_used or not ingredient2.is_used:
                return True
            pot1 = ingredient1.pot_id
            pot2 = ingredient2.pot_id
            r1 = self._scene_spec.pots[pot1].radius
            r2 = self._scene_spec.pots[pot2].radius
            dist = np.linalg.norm(np.array(ingredient1.pos) - np.array(ingredient2.pos))
            return dist >= r1 + r2

        for i, ingredient1 in enumerate(ingredient_variables[:-1]):
            for ingredient2 in ingredient_variables[i + 1 :]:
                constraint = FunctionalCSPConstraint(
                    f"{ingredient1.name}-{ingredient2.name}-nonoverlapping",
                    [ingredient1, ingredient2],
                    _pots_nonoverlapping,
                )
                constraints.append(constraint)

        # Ingredients cannot be in the same pot.
        def _ingredients_different_pots(
            ingredient1: _IngredientCSPState, ingredient2: _IngredientCSPState
        ) -> bool:
            # Unused pots need not be checked.
            if not ingredient1.is_used or not ingredient2.is_used:
                return True
            return ingredient1.pot_id != ingredient2.pot_id

        for i, ingredient1 in enumerate(ingredient_variables[:-1]):
            for ingredient2 in ingredient_variables[i + 1 :]:
                constraint = FunctionalCSPConstraint(
                    f"{ingredient1.name}-{ingredient2.name}-different-pots",
                    [ingredient1, ingredient2],
                    _ingredients_different_pots,
                )
                constraints.append(constraint)

        # Ingredient quantity used must be not more than total available.
        def _ingredient_quantity_exists(ingredient: _IngredientCSPState) -> bool:
            available = obs.ingredients[ingredient.name].ingredient_unused_quantity
            return ingredient.quantity <= available

        for ingredient in ingredient_variables:
            constraint = FunctionalCSPConstraint(
                f"{ingredient.name}-quantity-exists",
                [ingredient],
                _ingredient_quantity_exists,
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
        
        ingredient_variables = csp.variables[:-1]
        cooking_time_variable = csp.variables[-1]

        # Sample ingredients by sampling a meal from the meal model.
        def _sample_ingredients(
            sol: dict[CSPVariable, Any], rng: np.random.Generator
        ) -> dict[CSPVariable, Any]:
            # Sample a meal.
            meal = self._meal_model.sample(rng)
            # First determine the total amount of cooking time needed.
            total_cooking_time = meal.calculate_total_cooking_time(self._scene_spec)
            new_sol = {cooking_time_variable: total_cooking_time}
            # Now determine the start times for individual ingredients.
            for v in ingredient_variables:
                # Determine the temperature and quantity necessary for the meal.
                temp, quant = meal.ingredients[v.name]
                # Determine the cooking start time from the temperature.
                heat_rate = self._scene_spec.get_ingredient(v.name).heat_rate
                cooking_duration = int(np.round(temp / heat_rate))
                start_time = total_cooking_time - cooking_duration
                # Don't change positions, pots, etc.
                old_ing_state = sol[v]
                assert isinstance(old_ing_state, _IngredientCSPState)
                if v.name in meal.ingredients:
                    ing_state = old_ing_state.copy_with(is_used=True, quantity=quant, start_time=start_time)
                else:
                    ing_state = old_ing_state.copy_with(is_used=False)
                new_sol[v] = ing_state
            return new_sol

        ingredient_sampler = FunctionalCSPSampler(
            _sample_ingredients, csp, set(csp.variables)
        )

        # Sample pots and positions for any used ingredients.
        def _sample_pots(
            sol: dict[CSPVariable, Any], rng: np.random.Generator
        ) -> dict[CSPVariable, Any]:
            unused_pot_ids = list(range(len(obs.pots)))
            new_sol = {}
            for v in ingredient_variables:
                old_ing_state = sol[v]
                assert isinstance(old_ing_state, _IngredientCSPState)
                if old_ing_state.is_used:
                    pot_id = rng.choice(unused_pot_ids)
                    unused_pot_ids.remove(pot_id)
                    pot_pos_x = rng.uniform(0, self._scene_spec.stove_top_width)
                    pot_pos_y = rng.uniform(0, self._scene_spec.stove_top_height)
                    pot_pos = (pot_pos_x, pot_pos_y)
                else:
                    pot_id = -1
                    pot_pos = (-1, -1)
                ing_state = old_ing_state.copy_with(pot_id=pot_id, pos=pot_pos)
                new_sol[v] = ing_state
            return new_sol

        pot_position_sampler = FunctionalCSPSampler(
            _sample_pots, csp, set(csp.variables)
        )

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
        import ipdb

        ipdb.set_trace()

    def check_termination(self, obs: CookingState) -> bool:
        import ipdb

        ipdb.set_trace()
