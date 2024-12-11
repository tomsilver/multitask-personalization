"""A simple cooking environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, TypeAlias, get_args

import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame
from matplotlib import pyplot as plt

# NOTE: use intersection utilities in tomsgeoms2d once multiple pots are added.
from tomsgeoms2d.structs import Circle
from tomsutils.spaces import FunctionalSpace
from tomsutils.utils import fig2data

from multitask_personalization.structs import PublicSceneSpec


@dataclass(frozen=True)
class CookingPot:
    """A pot in a cooking environment."""

    radius: float = 0.25
    depth: float = 1.0

    position: tuple[float, float] | None = None


@dataclass(frozen=True)
class CookingIngredient:
    """An ingredient in a cooking environment."""

    name: str
    color: tuple[float, float, float] = (0.5, 0.5, 0.5)  # rendering
    initial_quantity: float = 1.0  # in terms of pot volume
    initial_temperature: float = 0.0  # heats up during cooking
    temperature_increase_rate: float = 0.1  # delta temperature


@dataclass(frozen=True)
class CookingIngredientPreference:
    """A user's preference for an ingredient in a cooking environment."""

    name: str
    min_quantity: float
    max_quantity: float


@dataclass(frozen=True)
class CookingSceneSpec(PublicSceneSpec):
    """Public parameters that define a cooking environment scene."""

    # The "stove top", a 2D rectangle. This is the only space in the env.
    stove_top_width: float = 10.0
    stove_top_height: float = 10.0

    # Characteristics of available pots.
    pots: list[CookingPot] = field(
        default_factory=lambda: [
            CookingPot(radius=0.5, depth=5.0, position=None),
            CookingPot(radius=1.0, depth=0.2, position=None),
            CookingPot(radius=0.5, depth=0.2, position=None),
            CookingPot(radius=0.5, depth=2.0, position=None),
        ]
    )

    # Characteristics of available ingredients.
    ingredients: Dict[str, CookingIngredient] = field(
        default_factory=lambda: {
            "salt": CookingIngredient(
                name="salt",
                color=(0.9, 0.9, 0.9),
                initial_quantity=0.5,
                initial_temperature=0.0,
                temperature_increase_rate=0.5,
            ),
            "pepper": CookingIngredient(
                name="pepper",
                color=(0.0, 0.0, 0.0),
                initial_quantity=1.0,
                initial_temperature=0.0,
                temperature_increase_rate=0.5,
            ),
            "sugar": CookingIngredient(
                name="sugar",
                color=(0.5, 0.0, 0.0),
                initial_quantity=1.0,
                initial_temperature=0.0,
                temperature_increase_rate=0.5,
            ),
            "flour": CookingIngredient(
                name="flour",
                color=(0.0, 0.5, 0.0),
                initial_quantity=20.0,
                initial_temperature=0.0,
                temperature_increase_rate=0.1,
            ),
        }
    )

    cooked_temperature: float = 10.0
    meal_cooling_rate: float = 0.1

    # Rendering.
    render_figscale: float = 5
    render_padding: float = 0.1


@dataclass(frozen=True)
class CookingHiddenSpec:
    """Hidden parameters for a cooking environment."""

    # Meal preferences.
    ingredient_preferences: Dict[str, CookingIngredientPreference] = field(
        default_factory=lambda: {
            "salt": CookingIngredientPreference(
                name="salt", min_quantity=0.1, max_quantity=0.2
            ),
            "pepper": CookingIngredientPreference(
                name="pepper", min_quantity=0.2, max_quantity=0.3
            ),
            "sugar": CookingIngredientPreference(
                name="sugar", min_quantity=0.0, max_quantity=0.1
            ),
            "flour": CookingIngredientPreference(
                name="flour", min_quantity=1.0, max_quantity=1.2
            ),
        }
    )

    # The ideal temperature range for food to be served.
    min_temperature: float = 0.4
    max_tempearture: float = 0.6


@dataclass(frozen=True)
class CookingPotState:
    """The state of a pot in a cooking environment."""

    # The position of the pot: on the stove top or not (None).
    position: tuple[float, float] | None

    # Ingredient in pot.
    ingredient_in_pot: str | None
    ingredient_quantity_in_pot: float
    ingredient_in_pot_temperature: float
    ingredient_done: bool

    def copy_with(
        self,
        position: tuple[float, float] | None = None,
        ingredient_in_pot: str | None = None,
        ingredient_quantity_in_pot: float | None = None,
        ingredient_in_pot_temperature: float | None = None,
        ingredient_done: bool | None = None,
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
            ingredient_done=(
                ingredient_done if ingredient_done is not None else self.ingredient_done
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
    ingredients: Dict[str, CookingIngredientState]

    meal_temperature: float | None


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
class CompleteCookingAction:
    """Signal that the cooking is complete."""


@dataclass(frozen=True)
class ServeMealCookingAction:
    """Implicitly combine all ingredients in all pots and serve the meal."""


CookingAction: TypeAlias = (
    MovePotCookingAction
    | AddIngredientCookingAction
    | WaitCookingAction
    | CompleteCookingAction
    | ServeMealCookingAction
)


class CookingEnv(gym.Env[CookingState, CookingAction]):
    """A simple cooking environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 2}

    def __init__(
        self,
        scene_spec: CookingSceneSpec,
        hidden_spec: CookingHiddenSpec | None = None,
        seed: int = 0,
    ) -> None:

        self._rng = np.random.default_rng(seed)
        self._hidden_spec = hidden_spec

        self.scene_spec = scene_spec
        self.render_mode = "rgb_array"
        self.action_space = FunctionalSpace(
            contains_fn=lambda x: isinstance(x, get_args(CookingAction)),
        )

        # Initialize state from scene spec.
        self._current_state = self._get_state_from_scene_spec(scene_spec)

        self._current_user_satisfaction = 0.0

    def _get_state_from_scene_spec(self, scene_spec: CookingSceneSpec) -> CookingState:
        return CookingState(
            pots=[
                CookingPotState(
                    position=pot.position,
                    ingredient_in_pot=None,
                    ingredient_quantity_in_pot=0.0,
                    ingredient_in_pot_temperature=0.0,
                    ingredient_done=False,
                )
                for pot in scene_spec.pots
            ],
            ingredients={
                ingredient_name: CookingIngredientState(
                    ingredient_unused_quantity=ingredient.initial_quantity
                )
                for ingredient_name, ingredient in scene_spec.ingredients.items()
            },
            meal_temperature=None,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[CookingState, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        # Reset the current state based on the scene spec.
        self._current_state = self._get_state_from_scene_spec(self.scene_spec)
        self._current_user_satisfaction = 0.0
        return self._get_state(), self._get_info()

    def step(
        self, action: CookingAction
    ) -> tuple[CookingState, float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action)

        # May be updated if the action is serve.
        self._current_user_satisfaction = 0.0

        # Heat up all ingredients in the pots if the pots are on the stove.
        for pot_id in range(len(self._current_state.pots)):
            pot = self._current_state.pots[pot_id]
            if pot.position is not None and pot.ingredient_in_pot is not None:
                new_temperature = (
                    pot.ingredient_in_pot_temperature
                    + self.scene_spec.ingredients[
                        pot.ingredient_in_pot
                    ].temperature_increase_rate
                )
                # Mark the ingredient as cooked if it has reached the cooked temperature.
                cooked = False
                if new_temperature > self.scene_spec.cooked_temperature or np.isclose(
                    new_temperature, self.scene_spec.cooked_temperature
                ):
                    cooked = True
                self._current_state.pots[pot_id] = pot.copy_with(
                    ingredient_in_pot_temperature=new_temperature,
                    ingredient_done=cooked,
                )
        # If cooking is complete, decrease the temperature of the meal.
        if self._current_state.meal_temperature is not None:
            self._current_state.meal_temperature = (
                self._current_state.meal_temperature - self.scene_spec.meal_cooling_rate
            )

        # Apply actions.
        if isinstance(action, MovePotCookingAction):
            # Should not move pots if cooking is complete.
            if self._current_state.meal_temperature is not None:
                raise ValueError("Cannot move pots after cooking is complete.")
            # Check that the position is within the stove top and is not occupied.
            if action.new_pot_position is not None:
                if (
                    action.new_pot_position[0]
                    - self.scene_spec.pots[action.pot_id].radius
                    < 0
                    or action.new_pot_position[0]
                    + self.scene_spec.pots[action.pot_id].radius
                    > self.scene_spec.stove_top_width
                    or action.new_pot_position[1]
                    - self.scene_spec.pots[action.pot_id].radius
                    < 0
                    or action.new_pot_position[1]
                    + self.scene_spec.pots[action.pot_id].radius
                    > self.scene_spec.stove_top_height
                ):
                    raise ValueError("Pot position is outside the stove top.")
                if any(
                    np.linalg.norm(
                        np.array(action.new_pot_position)
                        - np.array(self._current_state.pots[pot_id].position)
                    )
                    < self.scene_spec.pots[pot_id].radius
                    + self.scene_spec.pots[action.pot_id].radius
                    for pot_id in range(len(self._current_state.pots))
                    if self._current_state.pots[pot_id].position is not None
                    and pot_id != action.pot_id
                ):
                    raise ValueError("Pot position is already occupied.")
            # Move the pot to a new position (or off the stove).
            self._current_state.pots[action.pot_id] = self._current_state.pots[
                action.pot_id
            ].copy_with(position=action.new_pot_position)

        elif isinstance(action, AddIngredientCookingAction):
            # Should not add ingredients if cooking is complete.
            if self._current_state.meal_temperature is not None:
                raise ValueError("Cannot add ingredients after cooking is complete.")
            # Add the specified quantity of the ingredient into the pot.
            if self._current_state.pots[action.pot_id].position is None:
                raise ValueError(
                    "Cannot add ingredient to a pot that is not on the stove."
                )
            if self._current_state.pots[action.pot_id].ingredient_in_pot is not None:
                raise ValueError("Can only add ingredients to empty pots.")
            if action.ingredient not in self._current_state.ingredients:
                raise ValueError(f"Ingredient {action.ingredient} not supported.")
            if (
                action.ingredient_quantity
                > self._current_state.ingredients[
                    action.ingredient
                ].ingredient_unused_quantity
            ):
                raise ValueError("Not enough unused ingredient to add.")

            total_quantity = action.ingredient_quantity
            pot_volume = (
                np.pi
                * self.scene_spec.pots[action.pot_id].radius ** 2
                * self.scene_spec.pots[action.pot_id].depth
            )
            if total_quantity > pot_volume:
                raise ValueError("Cannot exceed the pot's capacity.")
            unused_quantity = (
                self._current_state.ingredients[
                    action.ingredient
                ].ingredient_unused_quantity
                - action.ingredient_quantity
            )
            # Update ingredient state.
            self._current_state.ingredients[action.ingredient] = (
                self._current_state.ingredients[action.ingredient].copy_with(
                    ingredient_unused_quantity=unused_quantity
                )
            )
            # Update pot state.
            self._current_state.pots[action.pot_id] = self._current_state.pots[
                action.pot_id
            ].copy_with(
                ingredient_in_pot=action.ingredient,
                ingredient_quantity_in_pot=total_quantity,
                ingredient_in_pot_temperature=self.scene_spec.ingredients[
                    action.ingredient
                ].initial_temperature,
            )

        elif isinstance(action, CompleteCookingAction):
            # Set meal temperature to cooked temperature.
            if self._current_state.meal_temperature is not None:
                raise ValueError("Meal temperature already set.")
            # Check if all ingredients in pots have been cooked.
            if not all(
                pot.ingredient_done
                for pot in self._current_state.pots
                if pot.ingredient_in_pot is not None
            ):
                raise ValueError("Cannot complete cooking with uncooked ingredients.")
            self._current_state.meal_temperature = self.scene_spec.cooked_temperature

        elif isinstance(action, WaitCookingAction):
            # Do nothing else.
            pass

        elif isinstance(action, ServeMealCookingAction):
            # Serve the meal and compute user satisfaction.
            if self._hidden_spec is None:
                raise ValueError("Hidden spec required for step().")
            # Check that all ingredients in pots have been cooked.
            if not all(
                pot.ingredient_done
                for pot in self._current_state.pots
                if pot.ingredient_in_pot is not None
            ):
                raise ValueError("Cannot serve a meal with uncooked ingredients.")
            # Check each ingredient quantity preferences.
            is_ingredient_correct = True
            ingredient_quantities = {
                ingredient: 0.0
                for ingredient in self._hidden_spec.ingredient_preferences.keys()
            }
            for pot in self._current_state.pots:
                if pot.ingredient_in_pot is not None:
                    ingredient_quantities[
                        pot.ingredient_in_pot
                    ] += pot.ingredient_quantity_in_pot
            for ingredient, ingredient_quantity in ingredient_quantities.items():
                is_quantity_correct = (
                    self._hidden_spec.ingredient_preferences[ingredient].min_quantity
                    <= ingredient_quantity
                    <= self._hidden_spec.ingredient_preferences[ingredient].max_quantity
                )
                if not is_quantity_correct:
                    is_ingredient_correct = False
                    break
            # Check the temperature of the meal.
            if self._current_state.meal_temperature is None:
                raise ValueError("Meal temperature not set.")
            is_temperature_correct = (
                self._hidden_spec.min_temperature
                <= self._current_state.meal_temperature
                <= self._hidden_spec.max_tempearture
            )

            if is_ingredient_correct and is_temperature_correct:
                self._current_user_satisfaction = 1.0
            else:
                self._current_user_satisfaction = -1.0

            # Reset the pot and ingredients.
            self._current_state = self._get_state_from_scene_spec(self.scene_spec)
        else:
            raise NotImplementedError()

        return self._get_state(), 0.0, False, False, self._get_info()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pad = self.scene_spec.render_padding
        min_x, max_x = -pad, self.scene_spec.stove_top_width + pad
        min_y, max_y = -pad, self.scene_spec.stove_top_height + pad

        scale = self.scene_spec.render_figscale
        fig, ax = plt.subplots(
            1, 1, figsize=(scale * (max_x - min_x), scale * (max_y - min_y))
        )
        # Plot pots.
        for pot_id, pot in enumerate(self._current_state.pots):
            if pot.position is not None:
                color = (
                    (1, 1, 1)
                    if pot.ingredient_in_pot is None
                    else self.scene_spec.ingredients[pot.ingredient_in_pot].color
                )
                circ = Circle(
                    pot.position[0],
                    pot.position[1],
                    self.scene_spec.pots[pot_id].radius,
                )
                circ.plot(ax, facecolor=color, edgecolor="black")
                # Plot ingredient name and quantity and temperature.
                if pot.ingredient_in_pot is not None:
                    # Plot text with white background.
                    ax.text(
                        pot.position[0],
                        pot.position[1],
                        f"{pot_id}\n"
                        + f"{pot.ingredient_in_pot}\n"
                        + f"num: {pot.ingredient_quantity_in_pot:.2f}\n"
                        + f"temp: {pot.ingredient_in_pot_temperature:.2f}\n"
                        + f"done: {pot.ingredient_done}",
                        ha="center",
                        va="center",
                        fontsize=50,
                        bbox={"facecolor": "white"},
                    )
        # Plot meal temperature
        if self._current_state.meal_temperature is not None:
            ax.text(
                max_x - pad,
                max_y - pad,
                f"meal temp: {self._current_state.meal_temperature:.2f}",
                ha="right",
                va="top",
                fontsize=100,
            )

        ax.set_xlim(min_x + pad, max_x - pad)
        ax.set_ylim(min_y + pad, max_y - pad)

        plt.tight_layout()

        img = fig2data(fig)
        plt.close()

        return img  # type: ignore

    def _get_state(self) -> CookingState:
        return self._current_state

    def _get_info(self) -> dict[str, Any]:
        return {
            "user_satisfaction": self._current_user_satisfaction,
        }
