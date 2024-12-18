"""A simple cooking environment."""

from __future__ import annotations

from typing import Any, get_args

import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame
from matplotlib import pyplot as plt
from tomsgeoms2d.structs import Circle
from tomsutils.spaces import FunctionalSpace
from tomsutils.utils import fig2data

from multitask_personalization.envs.cooking.cooking_hidden_spec import (
    CookingHappyMeal,
    CookingHiddenSpec,
)
from multitask_personalization.envs.cooking.cooking_scene_spec import CookingSceneSpec
from multitask_personalization.envs.cooking.cooking_structs import (
    AddIngredientCookingAction,
    CookingAction,
    CookingIngredientState,
    CookingPotState,
    CookingState,
    MovePotCookingAction,
    ServeMealCookingAction,
    ToggleStove,
    WaitCookingAction,
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
        # NOTE: the initial quantities of ingredients are randomized.
        return CookingState(
            stove_on=False,
            pots=[
                CookingPotState(
                    position=pot.position,
                    ingredient_in_pot=None,
                    ingredient_quantity_in_pot=0.0,
                    ingredient_in_pot_temperature=0.0,
                )
                for pot in scene_spec.pots
            ],
            ingredients={
                ingredient.name: CookingIngredientState(
                    ingredient_unused_quantity=self._rng.uniform(
                        *ingredient.respawn_quantity_bounds
                    ),
                )
                for ingredient in scene_spec.ingredients
            },
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
        done = False

        new_stove_on = self._current_state.stove_on
        new_pot_states: list[CookingPotState] = []
        new_ingredients = self._current_state.ingredients.copy()

        # Update pot temperatures and initialize new_pot_states.
        for pot_id, pot_state in enumerate(self._current_state.pots):
            old_temperature = pot_state.ingredient_in_pot_temperature
            # Only change temperature for pots with ingredients.
            if pot_state.ingredient_in_pot is not None:
                ingredient_spec = self.scene_spec.get_ingredient(
                    pot_state.ingredient_in_pot
                )
                # Increase temperature if on stove and stove is on.
                if pot_state.position is not None and self._current_state.stove_on:
                    assert ingredient_spec.heat_rate >= 0
                    temperature_change = ingredient_spec.heat_rate
                # Decrease temperature if off stove.
                else:
                    assert ingredient_spec.cool_rate >= 0
                    temperature_change = -1 * ingredient_spec.cool_rate
                new_temperature = old_temperature + temperature_change
            else:
                new_temperature = old_temperature
            # Note that the states will be modified further by the actions below.
            new_pot_state = pot_state.copy_with(
                ingredient_in_pot_temperature=new_temperature
            )
            new_pot_states.append(new_pot_state)

        # Handle move actions.
        if isinstance(action, MovePotCookingAction):
            pot_id = action.pot_id
            old_position = self._current_state.pots[pot_id].position
            # Always allowed to move off the stove.
            if action.new_pot_position is None:
                new_position = None
            # Check if move is valid: in bounds and not in collision.
            else:
                if self._pot_move_is_valid(
                    action.new_pot_position, pot_id, self._current_state
                ):
                    new_position = action.new_pot_position
                # Do nothing if action is not valid.
                else:
                    new_position = old_position
            # Finalize position.
            new_pot_states[pot_id] = new_pot_states[pot_id].copy_with(
                position=new_position
            )

        # Handle ingredient -> pot actions.
        elif isinstance(action, AddIngredientCookingAction):
            pot_id = action.pot_id
            ingredient = action.ingredient
            ingredient_quantity = action.ingredient_quantity
            if self._ingredient_add_is_valid(
                pot_id, ingredient, ingredient_quantity, self._current_state
            ):
                new_pot_states[pot_id] = new_pot_states[pot_id].copy_with(
                    ingredient_in_pot=ingredient,
                    ingredient_quantity_in_pot=ingredient_quantity,
                    ingredient_in_pot_temperature=0.0,
                )
                ingredient_state = self._current_state.ingredients[ingredient]
                remainder = (
                    ingredient_state.ingredient_unused_quantity - ingredient_quantity
                )
                new_ingredients[ingredient] = CookingIngredientState(remainder)

        # Handle meal serving.
        elif isinstance(action, ServeMealCookingAction):
            # Serve the meal and compute user satisfaction.
            if self._hidden_spec is None:
                raise ValueError("Hidden spec required for step().")
            # Check each ingredient quantity preferences.
            user_happy = False
            for happy_meal in self._hidden_spec.happy_meals:
                if self._happy_meal_is_made(happy_meal, self._current_state):
                    user_happy = True
                    break
            if user_happy:
                self._current_user_satisfaction = 1.0
            else:
                self._current_user_satisfaction = -1.0
            # Reset the pot and ingredients.
            self._current_state = self._get_state_from_scene_spec(self.scene_spec)
            done = True  # used for eval

        # Wait.
        elif isinstance(action, WaitCookingAction):
            pass

        # Handle stove toggling.
        elif isinstance(action, ToggleStove):
            new_stove_on = not new_stove_on

        else:
            raise NotImplementedError()

        # Update state.
        self._current_state = CookingState(
            new_stove_on, new_pot_states, new_ingredients
        )

        return self._get_state(), 0.0, done, False, self._get_info()

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
            if pot.position is not None and pot.ingredient_in_pot is not None:
                color = (
                    (1, 1, 1)
                    if pot.ingredient_in_pot is None
                    else self.scene_spec.get_ingredient(pot.ingredient_in_pot).color
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
                        + f"temp: {pot.ingredient_in_pot_temperature:.2f}\n",
                        ha="center",
                        va="center",
                        fontsize=50,
                        bbox={"facecolor": "white"},
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

    def _pot_move_is_valid(
        self, new_position: tuple[float, float], pot_id: int, state: CookingState
    ) -> bool:
        pot_radius = self.scene_spec.pots[pot_id].radius
        if (
            new_position[0] - pot_radius < 0
            or new_position[0] + pot_radius > self.scene_spec.stove_top_width
            or new_position[1] - pot_radius < 0
            or new_position[1] + pot_radius > self.scene_spec.stove_top_height
        ):
            # Out of bounds.
            return False
        for other_pot_id, other_pot_state in enumerate(state.pots):
            if other_pot_id == pot_id or other_pot_state.position is None:
                continue
            other_pot_radius = self.scene_spec.pots[other_pot_id].radius
            if (
                np.linalg.norm(
                    np.array(new_position) - np.array(other_pot_state.position)
                )
                < other_pot_radius + pot_radius
            ):
                # In collision with another pot.
                return False
        # Valid move.
        return True

    def _ingredient_add_is_valid(
        self,
        pot_id: int,
        ingredient: str,
        ingredient_quantity: float,
        state: CookingState,
    ) -> bool:
        # Can only add ingredients to empty pots.
        if state.pots[pot_id].ingredient_in_pot is not None:
            return False
        # The quantity selected is more than the quantity available.
        unused_quantity = state.ingredients[ingredient].ingredient_unused_quantity
        if ingredient_quantity > unused_quantity:
            return False
        # Valid.
        return True

    def _happy_meal_is_made(
        self, happy_meal: CookingHappyMeal, state: CookingState
    ) -> bool:
        for name, (temp_lo, temp_hi), (quant_lo, quant_hi) in happy_meal.ingredients:
            if not self._check_happy_meal_ingredient(
                name, temp_lo, temp_hi, quant_lo, quant_hi, state
            ):
                return False
        return True

    def _check_happy_meal_ingredient(
        self,
        name: str,
        temp_lo: float,
        temp_hi: float,
        quant_lo: float,
        quant_hi: float,
        state: CookingState,
    ) -> bool:
        # For simplicity, we assume all of the ingredient within the given bounds
        # needs to be contained within one pot. We don't split across pots.
        for pot_state in state.pots:
            if pot_state.ingredient_in_pot != name:
                continue
            quantity = pot_state.ingredient_quantity_in_pot
            if not quant_lo <= quantity <= quant_hi:
                return False
            temperature = pot_state.ingredient_in_pot_temperature
            if not temp_lo <= temperature <= temp_hi:
                return False
            return True
        return False
