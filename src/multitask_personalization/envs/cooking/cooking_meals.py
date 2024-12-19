"""Meal data structures and functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


DEFAULT_MEAL_SPECS = [
    MealSpec(
        "seasoning",
        [
            ("salt", (2.5, 3.5), (0.9, 1.1)),
            ("pepper", (2.5, 3.5), (0.9, 1.1)),
        ],
    )
]
