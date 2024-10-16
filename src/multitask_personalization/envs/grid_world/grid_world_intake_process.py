"""Intake process for the grid world."""

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from tomsutils.spaces import EnumSpace

from multitask_personalization.envs.grid_world.grid_world_mdp import _GridState
from multitask_personalization.envs.intake_process import IntakeProcess
from multitask_personalization.structs import CategoricalDistribution


@dataclass(frozen=True)
class _RewardTypeQuestion:
    """Is the reward for this location task-specific?"""

    loc: _GridState

    def __lt__(self, other: Any) -> bool:
        return str(self) < str(other)


@dataclass(frozen=True)
class _RewardValueQuestion:
    """Is the reward for location1 greater than that for location2?"""

    loc1: _GridState
    loc2: _GridState

    def __lt__(self, other: Any) -> bool:
        return str(self) < str(other)


@dataclass(frozen=True)
class _CoinFlipQuestion:
    """Flip a certain coin and return the boolean response."""

    coin_id: int

    def __lt__(self, other: Any) -> bool:
        return str(self) < str(other)


_GridIntakeObs: TypeAlias = bool
_GridIntakeAction: TypeAlias = (
    _RewardTypeQuestion | _RewardValueQuestion | _CoinFlipQuestion
)


class GridIntakeProcess(IntakeProcess[_GridIntakeObs, _GridIntakeAction]):
    """Intake process for the grid world."""

    def __init__(
        self,
        grid: NDArray[np.uint8],
        terminal_rewards: dict[tuple[int, int], float],
        terminal_types: dict[tuple[int, int], str],
        coin_weights: list[float],
        horizon: int,
    ) -> None:
        self._grid = grid
        self._terminal_rewards = terminal_rewards
        self._terminal_types = terminal_types
        self._coin_weights = coin_weights
        self._horizon = horizon
        assert set(self._terminal_rewards) == set(self._terminal_types)
        assert all(v in ("specific", "agnostic") for v in self._terminal_types.values())

    @property
    def observation_space(self) -> EnumSpace[_GridIntakeObs]:
        return EnumSpace([True, False])

    @property
    def action_space(self) -> EnumSpace[_GridIntakeAction]:
        actions: list[_GridIntakeAction] = []

        # Add reward type questions.
        for loc in self._terminal_types:
            actions.append(_RewardTypeQuestion(loc))

        # Add reward value questions.
        for loc1 in self._terminal_types:
            for loc2 in self._terminal_types:
                actions.append(_RewardValueQuestion(loc1, loc2))

        # Add coin flip questions.
        for coin_id in range(len(self._coin_weights)):
            actions.append(_CoinFlipQuestion(coin_id))

        return EnumSpace(actions)

    @property
    def horizon(self) -> int:
        return self._horizon

    def get_observation_distribution(
        self,
        action: _GridIntakeAction,
    ) -> CategoricalDistribution[_GridIntakeObs]:
        if isinstance(action, _RewardTypeQuestion):
            loc = action.loc
            reward_type = self._terminal_types[loc]
            response = reward_type == "specific"
            return CategoricalDistribution({response: 1.0})

        if isinstance(action, _RewardValueQuestion):
            loc1 = action.loc1
            loc2 = action.loc2
            response = self._terminal_rewards[loc1] > self._terminal_rewards[loc2]
            return CategoricalDistribution({response: 1.0})

        if isinstance(action, _CoinFlipQuestion):
            coin_id = action.coin_id
            weight = self._coin_weights[coin_id]
            return CategoricalDistribution({True: weight, False: 1.0 - weight})

        raise NotImplementedError
