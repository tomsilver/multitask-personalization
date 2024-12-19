"""Hidden specification for the cooking environment."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from multitask_personalization.envs.cooking.cooking_meals import Meal, MealSpec


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
