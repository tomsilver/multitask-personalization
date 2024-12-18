"""Cooking environment structs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


class _NoChange:
    def __repr__(self):
        return "<NoChange>"


_NO_CHANGE = _NoChange()


@dataclass(frozen=True)
class CookingPotState:
    """The state of a pot in a cooking environment."""

    # The position of the pot: on the stove top or not (None).
    position: tuple[float, float] | None

    # Ingredient in pot.
    ingredient_in_pot: str | None
    ingredient_quantity_in_pot: float
    ingredient_in_pot_temperature: float

    def copy_with(
        self,
        position: tuple[float, float] | None | _NoChange = _NO_CHANGE,
        ingredient_in_pot: str | None | _NoChange = _NO_CHANGE,
        ingredient_quantity_in_pot: float | _NoChange = _NO_CHANGE,
        ingredient_in_pot_temperature: float | _NoChange = _NO_CHANGE,
    ) -> CookingPotState:
        """Return a copy of the state with the specified fields updated."""
        if isinstance(position, _NoChange):
            new_position = self.position
        else:
            new_position = position

        if isinstance(ingredient_in_pot, _NoChange):
            new_ingredient_in_pot = self.ingredient_in_pot
        else:
            new_ingredient_in_pot = ingredient_in_pot

        if isinstance(ingredient_quantity_in_pot, _NoChange):
            new_ingredient_quantity_in_pot = self.ingredient_quantity_in_pot
        else:
            new_ingredient_quantity_in_pot = ingredient_quantity_in_pot

        if isinstance(ingredient_in_pot_temperature, _NoChange):
            new_ingredient_in_pot_temperature = self.ingredient_in_pot_temperature
        else:
            new_ingredient_in_pot_temperature = ingredient_in_pot_temperature

        return CookingPotState(
            position=new_position,
            ingredient_in_pot=new_ingredient_in_pot,
            ingredient_quantity_in_pot=new_ingredient_quantity_in_pot,
            ingredient_in_pot_temperature=new_ingredient_in_pot_temperature,
        )


@dataclass(frozen=True)
class CookingIngredientState:
    """The state of an ingredient in a cooking environment."""

    # Ingredient not yet in pot.
    ingredient_unused_quantity: float

    def copy_with(
        self, ingredient_unused_quantity: float | None = None
    ) -> CookingIngredientState:
        """Return a copy of the state with the specified fields updated."""
        return CookingIngredientState(
            ingredient_unused_quantity=(
                ingredient_unused_quantity
                if ingredient_unused_quantity is not None
                else self.ingredient_unused_quantity
            )
        )


# Allow updating of pots and ingredients.
@dataclass(frozen=False)
class CookingState:
    """The state of a cooking environment."""

    pots: list[CookingPotState]
    ingredients: dict[str, CookingIngredientState]

    def get_meal(self) -> Meal:
        """Extract a meal from the current ingredients in pots."""
        ingredients = {}
        for pot_state in self.pots:
            if pot_state.ingredient_in_pot is not None:
                ingredient = pot_state.ingredient_in_pot
                temperature = pot_state.ingredient_in_pot_temperature
                quantity = pot_state.ingredient_quantity_in_pot
                ingredients[ingredient] = (temperature, quantity)
        return Meal(ingredients)


@dataclass(frozen=True)
class MovePotCookingAction:
    """Move a pot on or off the stove top."""

    pot_id: int
    new_pot_position: tuple[float, float] | None


@dataclass(frozen=True)
class AddIngredientCookingAction:
    """Add some quantity of an unused ingredient into a pot."""

    pot_id: int
    ingredient: str
    ingredient_quantity: float


@dataclass(frozen=True)
class MultiCookingAction:
    """Do multiple actions in the same time step."""

    actions: list[CookingAction]


@dataclass(frozen=True)
class WaitCookingAction:
    """Do nothing (sometimes necessary to wait for things to heat up)."""


@dataclass(frozen=True)
class ServeMealCookingAction:
    """Implicitly combine all ingredients in all pots and serve the meal."""


CookingAction: TypeAlias = (
    MovePotCookingAction
    | AddIngredientCookingAction
    | WaitCookingAction
    | ServeMealCookingAction
    | MultiCookingAction
)


@dataclass(frozen=True)
class Meal:
    """A convenience data structure."""

    # Maps ingredient names to temperature and quantity.
    ingredients: dict[str, tuple[float, float]]
