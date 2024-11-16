"""A simple cooking environment."""

from dataclasses import dataclass
from typing import TypeAlias, get_args

import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame
from tomsutils.spaces import FunctionalSpace

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
    initial_ingredient_quantity: float = 1.0  # in terms of pot volume
    initial_ingredient_temperature: float = 0.0  # heats up during cooking
    ingredient_temperature_increase_rate: float = 0.1  # delta temperature


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

    def __init__(
        self,
        scene_spec: CookingSceneSpec,
        hidden_spec: CookingHiddenSpec | None = None,
        seed: int = 0,
    ) -> None:

        self._rng = np.random.default_rng(seed)
        self._scene_spec = scene_spec
        self._hidden_spec = hidden_spec

        self.action_space = FunctionalSpace(
            contains_fn=lambda x: isinstance(x, get_args(CookingAction)), seed=seed
        )
