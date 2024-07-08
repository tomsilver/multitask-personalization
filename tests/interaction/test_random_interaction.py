"""Tests for random_interaction.py."""

import numpy as np

from multitask_personalization.envs.grid_world import (
    _EMPTY,
    _OBSTACLE,
    GridTask,
)
from multitask_personalization.interaction.random_interaction import (
    RandomInteractionMethod,
)


def test_random_interaction():
    """Tests for random_interaction.py."""
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
    env = GridTask(
        "task0",
        grid,
        terminal_rewards,
        initial_state,
        terminal_types,
        coin_weights,
        horizon,
    )
    ip = env.intake_process
    im = RandomInteractionMethod(ip.action_space, ip.observation_space, seed=123)
    action = im.get_action()
    assert action in ip.action_space
    rng = np.random.default_rng(123)
    obs = ip.sample_next_observation(action, rng)
    im.observe(obs)
