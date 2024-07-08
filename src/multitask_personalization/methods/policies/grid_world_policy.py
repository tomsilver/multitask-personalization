"""A domain-specific parameterized policy for the grid world."""

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from multitask_personalization.envs.grid_world import _OBSTACLE, _GridAction, _GridState
from multitask_personalization.methods.policies.parameterized_policy import (
    ParameterizedPolicy,
)


class GridWorldParameterizedPolicy(
    ParameterizedPolicy[_GridState, _GridAction, _GridState]
):
    """A domain-specific parameterized policy for the grid world."""

    def __init__(self, grid: NDArray[np.uint8], terminal_locs: set[_GridState]) -> None:
        super().__init__()
        self._grid = grid
        self._terminal_locs = terminal_locs
        # Find all shortest paths to terminal locs.
        self._terminal_loc_tabular_policy = self._compute_tabular_policy()

    def step(self, state: _GridState) -> _GridAction:
        assert self._current_parameters is not None
        return self._terminal_loc_tabular_policy[self._current_parameters][state]

    def _compute_tabular_policy(
        self,
    ) -> dict[_GridState, dict[_GridState, str]]:

        # Set up conversion between (row, col) and state index.
        height, width = self._grid.shape
        num_states = height * width
        loc_to_idx: Callable[[_GridState], int] = lambda loc: loc[0] * width + loc[1]

        # Set up neighbors and actions.
        delta_to_action = {
            (-1, 0): "up",
            (1, 0): "down",
            (0, -1): "left",
            (0, 1): "right",
        }

        # Create adjacency matrix.
        neighbors = []
        for r in range(height):
            for c in range(width):
                if self._grid[r, c] == _OBSTACLE:
                    continue
                i = loc_to_idx((r, c))
                for dr, dc in delta_to_action:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if self._grid[nr, nc] != _OBSTACLE:
                        ni = loc_to_idx((nr, nc))
                        neighbors.append((i, ni))
        vals = np.ones(len(neighbors))
        nrows, ncols = zip(*neighbors)
        mat = coo_matrix((vals, (nrows, ncols)), shape=(num_states, num_states))

        # Solve all-pairs shortest paths.
        dist_matrix = shortest_path(mat)

        # Read out the tabular policy for each terminal loc.
        tabular_policy: dict[_GridState, dict[_GridState, str]] = {}
        for terminal_loc in self._terminal_locs:
            terminal_loc_idx = loc_to_idx(terminal_loc)
            tabular_policy_for_terminal: dict[_GridState, str] = {}
            for r in range(height):
                for c in range(width):
                    if self._grid[r, c] == _OBSTACLE or (r, c) == terminal_loc:
                        continue
                    i = loc_to_idx((r, c))
                    best_act: _GridAction | None = None
                    best_dist = np.inf
                    for (dr, dc), act in delta_to_action.items():
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < height and 0 <= nc < width):
                            continue
                        if self._grid[nr, nc] != _OBSTACLE:
                            j = loc_to_idx((nr, nc))
                            dist = dist_matrix[j, terminal_loc_idx]
                            if dist < best_dist:
                                best_dist = dist
                                best_act = act
                    assert best_act is not None
                    tabular_policy_for_terminal[(r, c)] = best_act
            tabular_policy[terminal_loc] = tabular_policy_for_terminal

        return tabular_policy
