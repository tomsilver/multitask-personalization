"""Tests for grid_mdp.py."""

import numpy as np

from multitask_personalization.envs.grid_mdp import _EMPTY, _OBSTACLE, GridMDP


def test_grid_mdp():
    """Tests for grid_mdp.py."""
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
    terminal_rewards = {
        (4, 0): 1,
        (4, 3): 10,
    }
    initial_state = (0, 0)
    mdp = GridMDP(grid, terminal_rewards, initial_state)
    assert not mdp.state_is_terminal(initial_state)
    assert mdp.state_is_terminal((4, 0))

    rng = np.random.default_rng(123)
    state = mdp.sample_initial_state(rng)
    ordered_actions = sorted(mdp.action_space)
    states = [state]
    for _ in range(5):
        action = rng.choice(ordered_actions)
        next_state = mdp.sample_next_state(state, action, rng)
        states.append(state)
        rew = mdp.get_reward(state, action, next_state)
        assert np.isclose(rew, 0.0)
        state = next_state

    # Uncomment for visualization.
    # import imageio.v2 as iio
    # imgs = [mdp.render_state(s) for s in states]
    # iio.mimsave("grid_mdp_test.gif", imgs)
