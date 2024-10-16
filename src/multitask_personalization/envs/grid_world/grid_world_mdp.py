"""A toy deterministic shortest-path MDP and intake process."""

from functools import cached_property
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from tomsutils.spaces import EnumSpace

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

    @cached_property
    def state_space(self) -> EnumSpace[_GridState]:
        return EnumSpace(
            [
                (r, c)
                for r in range(self._grid.shape[0])
                for c in range(self._grid.shape[1])
            ]
        )

    @cached_property
    def action_space(self) -> EnumSpace[_GridAction]:
        return EnumSpace(["up", "down", "left", "right"])

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
