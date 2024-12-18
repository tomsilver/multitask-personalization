"""Cooking environment structs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


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
        position: tuple[float, float] | None = None,
        ingredient_in_pot: str | None = None,
        ingredient_quantity_in_pot: float | None = None,
        ingredient_in_pot_temperature: float | None = None,
    ) -> CookingPotState:
        """Return a copy of the state with the specified fields updated."""
        return CookingPotState(
            position=position if position is not None else self.position,
            ingredient_in_pot=(
                ingredient_in_pot
                if ingredient_in_pot is not None
                else self.ingredient_in_pot
            ),
            ingredient_quantity_in_pot=(
                ingredient_quantity_in_pot
                if ingredient_quantity_in_pot is not None
                else self.ingredient_quantity_in_pot
            ),
            ingredient_in_pot_temperature=(
                ingredient_in_pot_temperature
                if ingredient_in_pot_temperature is not None
                else self.ingredient_in_pot_temperature
            ),
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
)
