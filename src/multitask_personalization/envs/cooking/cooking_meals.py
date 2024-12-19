"""Meal data structures and functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from multitask_personalization.envs.cooking.cooking_scene_spec import CookingSceneSpec


@dataclass(frozen=True)
class MealSpec:
    """A specification of one of the meals that the user enjoys."""

    # The unique name of the meal, e.g., "huevos rancheros".
    name: str

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
        return Meal(self.name, ingredients)

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


@dataclass(frozen=True)
class Meal:
    """A specific number of ingredients."""

    # From the meal spec.
    name: str

    # Maps ingredient names to temperature and quantity.
    ingredients: dict[str, tuple[float, float]]

    def calculate_total_cooking_time(self, scene_spec: CookingSceneSpec) -> int:
        """Calculate the total amount of time needed to cook this meal."""
        total_time = 0
        for ingredient, (temp, _) in self.ingredients.items():
            heat_rate = scene_spec.get_ingredient(ingredient).heat_rate
            # The plus 1 is to account for the one "add" step.
            ingredient_time = int(np.round(temp / heat_rate)) + 1
            total_time = max(total_time, ingredient_time)
        return total_time
