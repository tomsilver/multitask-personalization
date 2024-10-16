"""A toy deterministic shortest-path MDP."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from multitask_personalization.envs.grid_world.grid_world_intake_process import (
    GridIntakeProcess,
)
from multitask_personalization.envs.grid_world.grid_world_mdp import GridMDP
from multitask_personalization.envs.task import Task


@dataclass
class GridTask(Task):
    """The full grid world."""

    _id: str
    _grid: NDArray[np.uint8]
    _terminal_rewards: dict[tuple[int, int], float]
    _initial_state: tuple[int, int]
    _terminal_types: dict[tuple[int, int], str]
    _coin_weights: list[float]
    _intake_horizon: int

    @property
    def id(self) -> str:
        return self._id

    @property
    def mdp(self) -> GridMDP:
        return GridMDP(self._grid, self._terminal_rewards, self._initial_state)

    @property
    def intake_process(self) -> GridIntakeProcess:
        return GridIntakeProcess(
            self._grid,
            self._terminal_rewards,
            self._terminal_types,
            self._coin_weights,
            self._intake_horizon,
        )

    def close(self) -> None:
        pass
