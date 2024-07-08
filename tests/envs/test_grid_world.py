"""Tests for grid_world.py."""

import numpy as np

from multitask_personalization.envs.grid_world import (
    _EMPTY,
    _OBSTACLE,
    GridTask,
    _CoinFlipQuestion,
    _RewardTypeQuestion,
    _RewardValueQuestion,
)


def test_grid_world():
    """Tests for grid_world.py."""
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
    terminal_types = {
        (4, 0): "agnostic",
        (4, 3): "specific",
    }
    initial_state = (0, 0)
    coin_weights = [0.5, 1.0, 0.0]
    horizon = 5
    task = GridTask(
        "task0",
        grid,
        terminal_rewards,
        initial_state,
        terminal_types,
        coin_weights,
        horizon,
    )
    mdp = task.mdp
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

    terminal_types = {
        (4, 0): "agnostic",
        (4, 3): "specific",
    }
    coin_weights = [0.5, 1.0, 0.0]
    horizon = 5
    ip = task.intake_process
    assert ip.horizon == horizon
    assert ip.observation_space == {True, False}
    assert ip.action_space == {
        _RewardTypeQuestion((4, 0)),
        _RewardTypeQuestion((4, 3)),
        _RewardValueQuestion((4, 0), (4, 0)),
        _RewardValueQuestion((4, 0), (4, 3)),
        _RewardValueQuestion((4, 3), (4, 0)),
        _RewardValueQuestion((4, 3), (4, 3)),
        _CoinFlipQuestion(0),
        _CoinFlipQuestion(1),
        _CoinFlipQuestion(2),
    }
    dist = ip.get_observation_distribution(_RewardTypeQuestion((4, 0)))
    assert np.isclose(dist[False], 1.0)
    dist = ip.get_observation_distribution(_RewardTypeQuestion((4, 3)))
    assert np.isclose(dist[True], 1.0)
    dist = ip.get_observation_distribution(
        _RewardValueQuestion((4, 3), (4, 0)),
    )
    assert np.isclose(dist[True], 1.0)
    dist = ip.get_observation_distribution(
        _RewardValueQuestion((4, 0), (4, 0)),
    )
    assert np.isclose(dist[False], 1.0)
    dist = ip.get_observation_distribution(_CoinFlipQuestion(0))
    assert np.isclose(dist[True], 0.5)
