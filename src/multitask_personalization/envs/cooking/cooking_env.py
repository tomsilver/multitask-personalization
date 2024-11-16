"""A simple cooking environment."""

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
        return self._get_state(), self._get_info()

    def step(
        self, action: CookingAction
    ) -> tuple[CookingState, float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action)
        # TODO
        return self._get_state(), 0.0, False, False, self._get_info()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pad = self._scene_spec.render_padding
        min_x, max_x = -pad, self._scene_spec.stove_top_width + pad
        min_y, max_y = -pad, self._scene_spec.stove_top_height + pad

        scale = self._scene_spec.render_figscale
        fig, ax = plt.subplots(
            1, 1, figsize=(scale * (max_x - min_x), scale * (max_y - min_y))
        )

        # TODO
        if self._current_state.pot_position is not None:
            import ipdb

            ipdb.set_trace()

        ax.set_xlim(min_x + pad, max_x - pad)
        ax.set_ylim(min_y + pad, max_y - pad)

        plt.tight_layout()

        img = fig2data(fig)
        plt.close()

        return img

    def _get_state(self) -> CookingState:
        return self._current_state

    def _get_info(self) -> dict[str, Any]:
        return {}
