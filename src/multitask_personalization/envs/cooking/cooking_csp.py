"""CSP generation for the cooking environment."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

import numpy as np
from tomsutils.spaces import FunctionalSpace
from gymnasium.spaces import Discrete

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
            v: _IngredientCSPState(v.name, False, 0, 0, 0, (0.0, 0.0)) for v in ingredient_variables
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
            _: dict[CSPVariable, Any], rng: np.random.Generator
        ) -> dict[CSPVariable, Any]:
            meal = self._meal_model.sample(rng)
            var_to_state = {}
            for v in ingredient_variables:
                if v.name in meal.ingredients:
                    import ipdb; ipdb.set_trace()
                else:
                    var_to_state[v] = _IngredientCSPState(v.name, is_used=False, pot_id=0, start_time=0, pos=(0, 0))
            # TODO update cooking_time_variable
            return var_to_state

        ingredient_sampler = FunctionalCSPSampler(
            _sample_ingredients, csp, set(csp.variables)
        )

        # Sample positions for all currently used pots.
        def _sample_pot_positions(
            _: dict[CSPVariable, Any], rng: np.random.Generator
        ) -> dict[CSPVariable, Any]:
            import ipdb

            ipdb.set_trace()

        pot_position_sampler = FunctionalCSPSampler(
            _sample_pot_positions, csp, set(csp.variables)
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
