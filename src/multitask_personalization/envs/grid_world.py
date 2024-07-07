"""A toy deterministic shortest-path MDP and intake process."""

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from multitask_personalization.envs.intake_process import IntakeProcess
from multitask_personalization.envs.mdp import MDP
from multitask_personalization.structs import (
    CategoricalDistribution,
    Image,
)
from multitask_personalization.utils import render_avatar_grid

_GridState: TypeAlias = tuple[int, int]
_GridAction: TypeAlias = str  # up, down, left, right

# Grid cell types.
_EMPTY, _OBSTACLE = range(2)


class GridMDP(MDP[_GridState, _GridAction]):
    """A toy deterministic shortest-path MDP."""

    def __init__(
        self,
        grid: NDArray[np.uint8],
        terminal_rewards: dict[tuple[int, int], float],
        initial_state: tuple[int, int],
    ) -> None:
        self._grid = grid
        self._terminal_rewards = terminal_rewards
        self._initial_state = initial_state
        assert set(np.unique(self._grid)) <= {_EMPTY, _OBSTACLE}
        assert all(grid[r, c] == _EMPTY for r, c in self._terminal_rewards)
        assert self._grid[self._initial_state] == _EMPTY

    @property
    def state_space(self) -> set[_GridState]:
        return {
            (r, c)
            for r in range(self._grid.shape[0])
            for c in range(self._grid.shape[1])
        }

    @property
    def action_space(self) -> set[_GridAction]:
        return {"up", "down", "left", "right"}

    def state_is_terminal(self, state: _GridState) -> bool:
        return state in self._terminal_rewards

    def get_reward(
        self, state: _GridState, action: _GridAction, next_state: _GridState
    ) -> float:
        return self._terminal_rewards.get(next_state, 0.0)

    def get_initial_state_distribution(
        self,
    ) -> CategoricalDistribution[_GridState]:
        return CategoricalDistribution({self._initial_state: 1.0})

    def get_transition_distribution(
        self, state: _GridState, action: _GridAction
    ) -> CategoricalDistribution[_GridState]:
        act_to_delta = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }
        dr, dc = act_to_delta[action]
        r, c = state
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self._grid.shape[0] and 0 <= nc < self._grid.shape[1]):
            nr, nc = r, c
        if self._grid[nr, nc] == _OBSTACLE:
            nr, nc = r, c
        return CategoricalDistribution({(nr, nc): 1.0})

    def render_state(self, state: _GridState) -> Image:
        height, width = self._grid.shape
        avatar_grid = np.full((height, width), None, dtype=object)
        avatar_grid[state] = "robot"
        avatar_grid[self._grid == _OBSTACLE] = "obstacle"
        for r, c in self._terminal_rewards:
            avatar_grid[r, c] = "hidden"
        return render_avatar_grid(avatar_grid)


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
    """Flip a certain coin and return the boolean resposne."""

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
    def observation_space(self) -> set[_GridIntakeObs]:
        return {True, False}

    @property
    def action_space(self) -> set[_GridIntakeAction]:
        actions: set[_GridIntakeAction] = set()

        # Add reward type questions.
        for loc in self._terminal_types:
            actions.add(_RewardTypeQuestion(loc))

        # Add reward value questions.
        for loc1 in self._terminal_types:
            for loc2 in self._terminal_types:
                actions.add(_RewardValueQuestion(loc1, loc2))

        # Add coin flip questions.
        for coin_id in range(len(self._coin_weights)):
            actions.add(_CoinFlipQuestion(coin_id))

        return actions

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
