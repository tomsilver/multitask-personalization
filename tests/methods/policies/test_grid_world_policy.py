"""Tests for grid_world_policy.py."""

import numpy as np

from multitask_personalization.envs.grid_world import (
    _EMPTY,
    _OBSTACLE,
)
from multitask_personalization.methods.policies.grid_world_policy import (
    GridWorldParameterizedPolicy,
)


def test_grid_world_policy():
    """Tests for grid_world_policy.py."""
    E, O = _EMPTY, _OBSTACLE
    grid = np.array(
        [
            [E, E, E, E, E],
            [E, O, E, O, E],
            [O, O, E, E, E],
            [E, E, E, E, E],
            [E, E, O, E, E],
        ]
    )
    terminal_locs = {
        (4, 0),
        (4, 3),
    }
    policy = GridWorldParameterizedPolicy(grid, terminal_locs)
    policy.reset("task0", (4, 0))
    assert policy.step((3, 2)) == "left"
    policy.reset("task0", (4, 3))
    assert policy.step((3, 2)) == "right"
