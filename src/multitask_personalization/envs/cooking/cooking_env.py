"""A simple cooking environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias, get_args

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
class CookingSceneSpec(PublicSceneSpec):
    """Public parameters that define a cooking environment scene."""

    # The "stove top", a 2D rectangle. This is the only space in the env.
    stove_top_width: float = 1.0
    stove_top_height: float = 1.0

    # Characteristics of a pot. In the near future, generalize to multiple.
    # Assume that the pots are empty and not on the stove in the initial state.
    pot_radius: float = 0.25
    pot_depth: float = 1.0  # to define a volume for ingredient capacity

    # Characteristics of a single ingredient. In the near future, generalize
    # this to multiple ingredients.
    ingredient_name: str = "salt"
    ingredient_color: tuple[float, float, float] = (0.5, 0.5, 0.5)  # rendering
    initial_ingredient_quantity: float = 1.0  # in terms of pot volume
    initial_ingredient_temperature: float = 0.0  # heats up during cooking
    ingredient_temperature_increase_rate: float = 0.1  # delta temperature

    # Rendering.
    render_figscale: float = 5
    render_padding: float = 0.1


@dataclass(frozen=True)
class CookingHiddenSpec:
    """Hidden parameters for a cooking environment."""

    # Meal preferences. In the near future, figure out how to generalize this.
    # For now, the only preference is in terms of amount of salt.
    min_amount_salt: float = 0.1
    max_amount_salt: float = 0.2

    # The ideal temperature range for food to be served.
    min_temperature: float = 0.4
    max_tempearture: float = 0.6


@dataclass(frozen=True)
class CookingState:
    """The state of a cooking environment."""

    # The position of the pot: on the stove top or not (None).
    pot_position: tuple[float, float] | None

    # Ingredient in pot.
    ingredient_in_pot: str | None
    ingredient_quantity_in_pot: float
    ingredient_in_pot_temperature: float

    # Ingredient not yet in pot.
    ingredient_unused_quantity: float

    def copy_with(
        self,
        pot_position: tuple[float, float] | None = None,
        ingredient_in_pot: str | None = None,
        ingredient_quantity_in_pot: float | None = None,
        ingredient_in_pot_temperature: float | None = None,
        ingredient_unused_quantity: float | None = None,
    ) -> CookingState:
        """Return a new CookingState with updated fields."""
        return CookingState(
            pot_position=(
                pot_position if pot_position is not None else self.pot_position
            ),
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
            ingredient_unused_quantity=(
                ingredient_unused_quantity
                if ingredient_unused_quantity is not None
                else self.ingredient_unused_quantity
            ),
        )


@dataclass(frozen=True)
class MovePotCookingAction:
    """Move a pot on or off the stove top."""

    new_pot_position: tuple[float, float] | None


@dataclass(frozen=True)
class AddIngredientCookingAction:
    """Add some quantity of an unused ingredient into a pot."""

    ingredient_name: str
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
        self._scene_spec = scene_spec
        self._hidden_spec = hidden_spec

        self.render_mode = "rgb_array"
        self.action_space = FunctionalSpace(
            contains_fn=lambda x: isinstance(x, get_args(CookingAction)),
        )

        # Reset in reset().
        self._current_state = CookingState(
            pot_position=None,
            ingredient_in_pot=None,
            ingredient_quantity_in_pot=0.0,
            ingredient_in_pot_temperature=0.0,
            ingredient_unused_quantity=0.0,
        )
        self._current_user_satisfaction = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[CookingState, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        # Reset the current state based on the scene spec.
        self._current_state = CookingState(
            pot_position=None,
            ingredient_in_pot=None,
            ingredient_quantity_in_pot=0.0,
            ingredient_in_pot_temperature=0.0,
            ingredient_unused_quantity=self._scene_spec.initial_ingredient_quantity,
        )
        self._current_user_satisfaction = 0.0
        return self._get_state(), self._get_info()

    def step(
        self, action: CookingAction
    ) -> tuple[CookingState, float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action)

        # May be updated if the action is serve.
        self._current_user_satisfaction = 0.0

        # Heat up the ingredient in the pot if the pot is on the stove.
        if self._current_state.ingredient_in_pot is not None:
            new_temperature = (
                self._current_state.ingredient_in_pot_temperature
                + self._scene_spec.ingredient_temperature_increase_rate
            )
            self._current_state = self._current_state.copy_with(
                ingredient_in_pot_temperature=new_temperature
            )

        # Apply actions.
        if isinstance(action, MovePotCookingAction):
            # Move the pot to a new position (or off the stove).
            self._current_state = self._current_state.copy_with(
                pot_position=action.new_pot_position
            )

        elif isinstance(action, AddIngredientCookingAction):
            # Add the specified quantity of the ingredient into the pot.
            if self._current_state.pot_position is None:
                raise ValueError(
                    "Cannot add ingredient to a pot that is not on the stove."
                )
            if self._current_state.ingredient_in_pot is not None:
                raise ValueError("Can only add ingredients to empty pots.")
            if action.ingredient_name != self._scene_spec.ingredient_name:
                raise ValueError(f"Ingredient {action.ingredient_name} not supported.")
            if (
                action.ingredient_quantity
                > self._current_state.ingredient_unused_quantity
            ):
                raise ValueError("Not enough unused ingredient to add.")

            total_quantity = (
                self._current_state.ingredient_quantity_in_pot
                + action.ingredient_quantity
            )
            pot_volume = (
                2 * np.pi * self._scene_spec.pot_radius * self._scene_spec.pot_depth
            )
            if total_quantity > pot_volume:
                raise ValueError("Cannot exceed the pot's capacity.")
            unused_quantity = (
                self._current_state.ingredient_unused_quantity
                - action.ingredient_quantity
            )

            self._current_state = self._current_state.copy_with(
                ingredient_in_pot=action.ingredient_name,
                ingredient_quantity_in_pot=total_quantity,
                ingredient_unused_quantity=unused_quantity,
            )

        elif isinstance(action, WaitCookingAction):
            # Do nothing else.
            pass

        elif isinstance(action, ServeMealCookingAction):
            # Serve the meal and compute user satisfaction.
            if self._hidden_spec is None:
                raise ValueError("Hidden spec required for step().")

            # Check ingredient quantity and temperature preferences.
            is_salt_correct = (
                self._hidden_spec.min_amount_salt
                <= self._current_state.ingredient_quantity_in_pot
                <= self._hidden_spec.max_amount_salt
            )
            is_temperature_correct = (
                self._hidden_spec.min_temperature
                <= self._current_state.ingredient_in_pot_temperature
                <= self._hidden_spec.max_tempearture
            )

            if is_salt_correct and is_temperature_correct:
                self._current_user_satisfaction = 1.0
            else:
                self._current_user_satisfaction = -1.0

            # Reset the pot and ingredients.
            self._current_state = CookingState(
                pot_position=None,
                ingredient_in_pot=None,
                ingredient_quantity_in_pot=0.0,
                ingredient_in_pot_temperature=0.0,
                ingredient_unused_quantity=self._scene_spec.initial_ingredient_quantity,
            )
        else:
            raise NotImplementedError()

        return self._get_state(), 0.0, False, False, self._get_info()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pad = self._scene_spec.render_padding
        min_x, max_x = -pad, self._scene_spec.stove_top_width + pad
        min_y, max_y = -pad, self._scene_spec.stove_top_height + pad

        scale = self._scene_spec.render_figscale
        fig, ax = plt.subplots(
            1, 1, figsize=(scale * (max_x - min_x), scale * (max_y - min_y))
        )

        if self._current_state.pot_position is not None:
            color = (
                (1, 1, 1)
                if self._current_state.ingredient_in_pot is None
                else self._scene_spec.ingredient_color
            )
            circ = Circle(
                self._current_state.pot_position[0],
                self._current_state.pot_position[1],
                self._scene_spec.pot_radius,
            )
            circ.plot(ax, facecolor=color, edgecolor="black")

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
