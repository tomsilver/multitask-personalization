"""Tests for approach.py."""

import numpy as np

from multitask_personalization.envs.grid_world import (
    _EMPTY,
    _OBSTACLE,
    GridTask,
)
from multitask_personalization.methods.approach import Approach
from multitask_personalization.methods.calibration.grid_world_calibrator import (
    GridWorldCalibrator,
)
from multitask_personalization.methods.interaction.random_interaction import (
    RandomInteractionMethod,
)
from multitask_personalization.methods.policies.grid_world_policy import (
    GridWorldParameterizedPolicy,
)


def test_approach():
    """Tests for approach.py."""

    # Create the environment.
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
    horizon = 100
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
    ip = task.intake_process
    seed = 123

    # Create the approach.
    calibrator = GridWorldCalibrator(set(terminal_rewards))
    im = RandomInteractionMethod(seed)
    policy = GridWorldParameterizedPolicy(grid, set(terminal_rewards))
    approach = Approach(calibrator, im, policy)

    # Run the intake process.
    rng = np.random.default_rng(seed)
    approach.reset(task.id, ip.action_space, ip.observation_space)
    for _ in range(horizon):
        act = approach.get_intake_action()
        obs = ip.sample_next_observation(act, rng)
        approach.record_intake_observation(obs)
    approach.finish_intake()

    # Run the MDP; should get the maximal reward.
    rng = np.random.default_rng(seed)
    state = mdp.sample_initial_state(rng)
    rew = 0
    for _ in range(100):
        if mdp.state_is_terminal(state):
            break
        action = approach.get_mdp_action(state)
        next_state = mdp.sample_next_state(state, action, rng)
        rew += mdp.get_reward(state, action, next_state)
        state = next_state
    assert np.isclose(rew, 10.0)
