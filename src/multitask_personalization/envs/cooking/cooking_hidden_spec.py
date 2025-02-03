"""Hidden specification for the cooking environment."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from multitask_personalization.envs.cooking.cooking_meals import (
    Meal,
    MealSpec,
)
from multitask_personalization.envs.cooking.cooking_structs import IngredientCritique
from multitask_personalization.utils import Bounded1DClassifier


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

    @abc.abstractmethod
    def update(self, meal: Meal, critiques: list[IngredientCritique]) -> None:
        """Update the model from user feedback."""


class MealSpecMealPreferenceModel(MealPreferenceModel):
    """An explicit list of a user's meal preferences."""

    def __init__(self, meal_specs: list[MealSpec]) -> None:
        self._universal_meal_specs = {m.name: m for m in meal_specs}
        assert len(self._universal_meal_specs) == len(
            meal_specs
        ), "Meal names must be unique"
        # Initialize models for quantity and temperature for each meal ingredient.
        self._temperature_models: dict[str, dict[str, Bounded1DClassifier]] = {}
        self._quantity_models: dict[str, dict[str, Bounded1DClassifier]] = {}
        for meal_name, meal_spec in self._universal_meal_specs.items():
            self._temperature_models[meal_name] = {}
            self._quantity_models[meal_name] = {}
            for ing_spec in meal_spec.ingredients:
                temp_lo, temp_hi = ing_spec.temperature
                self._temperature_models[meal_name][ing_spec.name] = (
                    Bounded1DClassifier(temp_lo, temp_hi, default=1.0)
                )
                quant_lo, quant_hi = ing_spec.quantity
                self._quantity_models[meal_name][ing_spec.name] = Bounded1DClassifier(
                    quant_lo, quant_hi, default=1.0
                )

    def sample(self, rng: np.random.Generator) -> Meal:
        meal_spec_idx = rng.choice(len(self._universal_meal_specs))
        ordered_meal_specs = [
            self._universal_meal_specs[m] for m in sorted(self._universal_meal_specs)
        ]
        meal_spec = ordered_meal_specs[meal_spec_idx]
        return meal_spec.sample(rng)

    def meal_is_known(self, meal_name: str) -> bool:
        return meal_name in self._universal_meal_specs

    def predict_enjoyment_logprob(self, meal: Meal) -> float:
        if not meal.name in self._universal_meal_specs:
            return -np.inf
        # Accumulate estimates for the whole meal.
        meal_spec = self._universal_meal_specs[meal.name]
        total_log_prob = 0.0
        for ing_spec in meal_spec.ingredients:
            # If the ingredient is missing, fail.
            if ing_spec.name not in meal.ingredients:
                return -np.inf
            temp, quant = meal.ingredients[ing_spec.name]
            # Consider temperature.
            temperature_model = self._temperature_models[meal.name][ing_spec.name]
            temperature_log_prob = np.log(temperature_model.predict_proba([temp])[0])
            total_log_prob += temperature_log_prob
            # Consier quantity.
            quantity_model = self._quantity_models[meal.name][ing_spec.name]
            quantity_log_prob = np.log(quantity_model.predict_proba([quant])[0])
            total_log_prob += quantity_log_prob
        return total_log_prob

    def get_feedback(self, meal: Meal) -> list[IngredientCritique]:
        meal_spec = self._universal_meal_specs[meal.name]
        critiques: list[IngredientCritique] = []
        for ing_spec in meal_spec.ingredients:
            temperature_feedback = "good"
            quantity_feedback = "good"
            missing = False
            if ing_spec.name not in meal.ingredients:
                missing = True
            else:
                temp, quant = meal.ingredients[ing_spec.name]
                if temp < ing_spec.temperature[0]:
                    temperature_feedback = "hotter"
                elif temp > ing_spec.temperature[1]:
                    temperature_feedback = "colder"
                if quant < ing_spec.quantity[0]:
                    quantity_feedback = "more"
                elif quant > ing_spec.quantity[1]:
                    quantity_feedback = "less"
            if missing or temperature_feedback != "good" or quantity_feedback != "good":
                critique = IngredientCritique(
                    ing_spec.name, temperature_feedback, quantity_feedback
                )
                critiques.append(critique)
        return critiques

    def update(self, meal: Meal, critiques: list[IngredientCritique]) -> None:
        for critique in critiques:
            assert not critique.missing
            meal_temp, meal_quant = meal.ingredients[critique.ingredient]
            temperature_label = critique.hotter_or_colder == "good"
            quantity_label = critique.more_or_less == "good"
            temperature_model = self._temperature_models[meal.name][critique.ingredient]
            quantity_model = self._quantity_models[meal.name][critique.ingredient]
            temperature_model.fit_incremental([meal_temp], [temperature_label])
            quantity_model.fit_incremental([meal_quant], [quantity_label])
