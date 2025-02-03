"""Hidden specification for the cooking environment."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from multitask_personalization.envs.cooking.cooking_meals import Meal, MealSpec, IngredientSpec
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

    @abc.abstractmethod
    def update(self, meal: Meal, critiques: list[IngredientCritique]) -> None:
        """Update the model from user feedback."""


class MealSpecMealPreferenceModel(MealPreferenceModel):
    """An explicit list of a user's meal preferences."""

    def __init__(self, meal_specs: list[MealSpec]) -> None:
        self._meal_specs = {m.name: m for m in meal_specs}
        assert len(self._meal_specs) == len(meal_specs), "Meal names must be unique"

    def sample(self, rng: np.random.Generator) -> Meal:
        meal_spec_idx = rng.choice(len(self._meal_specs))
        ordered_meal_specs = [self._meal_specs[m] for m in sorted(self._meal_specs)]
        meal_spec = ordered_meal_specs[meal_spec_idx]
        return meal_spec.sample(rng)

    def meal_is_known(self, meal_name: str) -> bool:
        return meal_name in self._meal_specs

    def predict_enjoyment_logprob(self, meal: Meal) -> float:
        if not meal.name in self._meal_specs:
            return -np.inf
        enjoys = self._meal_specs[meal.name].check(meal)
        return 0.0 if enjoys else -np.inf

    def get_feedback(self, meal: Meal) -> list[IngredientCritique]:
        meal_spec = self._meal_specs[meal.name]
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
        old_ingredients = {i.name: i for i in self._meal_specs[meal.name].ingredients}
        new_ingredients = old_ingredients.copy()
        for critique in critiques:
            assert not critique.missing
            meal_temp, meal_quant = meal.ingredients[critique.ingredient]
            temp_lo, temp_hi = old_ingredients[critique.ingredient].temperature
            quant_lo, quant_hi = old_ingredients[critique.ingredient].quantity
            # Update temperature.
            if critique.hotter_or_colder == "hotter":
                temp_lo = meal_temp
            elif critique.hotter_or_colder == "colder":
                temp_hi = meal_temp
            else:
                assert critique.hotter_or_colder == "good"
            # Update quantity.
            if critique.more_or_less == "more":
                quant_lo = meal_quant
            elif critique.more_or_less == "less":
                quant_hi = meal_quant
            else:
                assert critique.more_or_less == "good"
            new_ingredients[critique.ingredient] = IngredientSpec(critique.ingredient,
                                                                  temperature=(temp_lo, temp_hi),
                                                                  quantity=(quant_lo, quant_hi))
        # Finalize new meal spec.
        self._meal_specs[meal.name] = MealSpec(
            name=meal.name,
            ingredients=list(new_ingredients.values())
        )
