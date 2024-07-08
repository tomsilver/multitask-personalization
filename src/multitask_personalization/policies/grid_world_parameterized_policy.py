"""A domain-specific parameterized policy for the grid world."""


from multitask_personalization.policies.parameterized_policy import ParameterizedPolicy
from multitask_personalization.envs.mdp import MDPState, MDPAction
from multitask_personalization.envs.grid_world import _OBSTACLE
from numpy.typing import NDArray
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import coo_matrix
import numpy as np


class GridWorldParameterizedPolicy(ParameterizedPolicy):
    """A domain-specific parameterized policy for the grid world."""

    def __init__(self, grid: NDArray[np.uint8], terminal_locs: set[tuple[int, int]]) -> None:
        super().__init__()
        self._grid = grid
        self._terminal_locs = terminal_locs
        # Find all shortest paths to terminal locs.
        self._terminal_loc_tabular_policy = self._compute_tabular_policy()

    def step(self, state: MDPState) -> MDPAction:
        assert self._current_parameters is not None
        return self._terminal_loc_tabular_policy[self._current_parameters][state]
    
    def _compute_tabular_policy(self) -> dict[tuple[int, int], dict[tuple[int, int], str]]:
        
        # Set up conversion between (row, col) and state index.
        height, width = self._grid.shape
        num_states = height * width
        loc_to_idx = lambda loc: loc[0] * width + loc[1]
        idx_to_loc = lambda idx: (idx // width, idx % width)

        # Create adjacency matrix.
        neighbors = []
        for r in range(height):
            for c in range(width):
                if self._grid[r, c] == _OBSTACLE:
                    continue
                i = loc_to_idx((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
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
        dist_matrix, predecessors = shortest_path(mat, return_predecessors=True)

        # Read out the tabular policy for each terminal loc.
        tabular_policy = dict[tuple[int, int], dict[tuple[int, int], str]]
        for terminal_loc in self._terminal_locs:
            import ipdb; ipdb.set_trace()

