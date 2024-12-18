"""Hidden specification for the cooking environment."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from multitask_personalization.envs.cooking.cooking_structs import Meal


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
    def check(self, meal: Meal) -> bool:
        """Check whether the user would enjoy this meal."""


@dataclass(frozen=True)
class MealSpec:
    """A specification of one of the meals that the user enjoys."""

    # Ingredient name, temperature bounds, quantity bounds.
    ingredients: list[tuple[str, tuple[float, float], tuple[float, float]]]

    def __post_init__(self) -> None:
        for _, (temp_lo, temp_hi), (quant_lo, quant_hi) in self.ingredients:
            assert 0 <= temp_lo < temp_hi
            assert 0 <= quant_lo < quant_hi

    def sample(self, rng: np.random.Generator) -> Meal:
        """Sample a meal."""
        ingredients = {}
        for ing, (temp_lo, temp_hi), (quant_lo, quant_hi) in self.ingredients:
            temp = rng.uniform(temp_lo, temp_hi)
            quant = rng.uniform(quant_lo, quant_hi)
            ingredients[ing] = (temp, quant)
        return Meal(ingredients)

    def check(self, meal: Meal) -> bool:
        """Check if a meal fits the preferences."""
        # For simplicity, we assume all of the ingredient within the given bounds
        # needs to be contained within one pot. We don't split across pots.
        for ing, (temp_lo, temp_hi), (quant_lo, quant_hi) in self.ingredients:
            if ing not in meal.ingredients:
                return False
            temp, quant = meal.ingredients[ing]
            if not temp_lo <= temp <= temp_hi:
                return False
            if not quant_lo <= quant <= quant_hi:
                return False
        return True


class MealSpecMealPreferenceModel(MealPreferenceModel):
    """An explicit list of a user's meal preferences."""

    def __init__(self, meal_specs: list[MealSpec]) -> None:
        self._meal_specs = meal_specs

    def sample(self, rng: np.random.Generator) -> Meal:
        meal_spec_idx = rng.choice(len(self._meal_specs))
        meal_spec = self._meal_specs[meal_spec_idx]
        return meal_spec.sample(rng)

    def check(self, meal: Meal) -> bool:
        return any(ms.check(meal) for ms in self._meal_specs)
