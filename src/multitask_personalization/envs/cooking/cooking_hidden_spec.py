"""Hidden specification for the cooking environment."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from multitask_personalization.envs.cooking.cooking_meals import Meal, MealSpec
from multitask_personalization.envs.cooking.cooking_structs import IngredientCritique


@dataclass(frozen=True)
class CookingHiddenSpec:
    """Hidden parameters for a cooking environment."""

    meal_preference_model: MealPreferenceModel


class MealPreferenceModel(abc.ABC):
    """A model of a user's meal preferences."""

    @abc.abstractmethod
    def sample(self, rng: np.random.Generator) -> Meal:
        """Sample a meal that the user should enjoy."""

    @abc.abstractmethod
    def meal_is_known(self, meal_name: str) -> bool:
        """Whether the name of the meal is known."""

    @abc.abstractmethod
    def predict_enjoyment_logprob(self, meal: Meal) -> float:
        """Predict the log probability of enjoyment."""

    @abc.abstractmethod
    def get_feedback(self, meal: Meal) -> list[IngredientCritique]:
        """Critique all the ingredients in this meal."""


class MealSpecMealPreferenceModel(MealPreferenceModel):
    """An explicit list of a user's meal preferences."""

    def __init__(self, meal_specs: list[MealSpec]) -> None:
        self._meal_specs = meal_specs

    def sample(self, rng: np.random.Generator) -> Meal:
        meal_spec_idx = rng.choice(len(self._meal_specs))
        meal_spec = self._meal_specs[meal_spec_idx]
        return meal_spec.sample(rng)

    def meal_is_known(self, meal_name: str) -> bool:
        for meal_spec in self._meal_specs:
            if meal_name == meal_spec.name:
                return True
        return False

    def predict_enjoyment_logprob(self, meal: Meal) -> float:
        # Degenerate.
        enjoys = any(ms.check(meal) for ms in self._meal_specs)
        return 0.0 if enjoys else -np.inf

    def get_feedback(self, meal: Meal) -> list[IngredientCritique]:
        meal_spec = self._get_spec_for_meal(meal.name)
        critiques: list[IngredientCritique] = []
        for ing, (temp_lo, temp_hi), (quant_lo, quant_hi) in meal_spec.ingredients:
            temperature_feedback = "good"
            quantity_feedback = "good"
            missing = False
            if ing not in meal.ingredients:
                missing = True
            else:
                temp, quant = meal.ingredients[ing]
                if temp < temp_lo:
                    temperature_feedback = "hotter"
                elif temp > temp_hi:
                    temperature_feedback = "cooler"
                if quant < quant_lo:
                    quantity_feedback = "more"
                elif quant > quant_hi:
                    quantity_feedback = "less"
            if missing or temperature_feedback != "good" or quantity_feedback != "good":
                critique = IngredientCritique(
                    ing, temperature_feedback, quantity_feedback
                )
                critiques.append(critique)
        return critiques

    def _get_spec_for_meal(self, meal_name: str) -> MealSpec:
        for meal_spec in self._meal_specs:
            if meal_name == meal_spec.name:
                return meal_spec
        raise ValueError(f"Unknown meal {meal_name}")
