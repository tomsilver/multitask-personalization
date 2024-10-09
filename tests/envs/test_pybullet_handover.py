"""Tests for pybullet_handover.py."""

import numpy as np

from multitask_personalization.envs.pybullet_handover import (
    PyBulletHandoverTask,
    _ROMReachableQuestion,
)


def test_pybullet_handover():
    """Tests for pybullet_handover.py."""
    intake_horizon = 5
    task = PyBulletHandoverTask(
        intake_horizon=intake_horizon,
        use_gui=True,
    )
    mdp = task.mdp
    rng = np.random.default_rng(123)
    state = mdp.sample_initial_state(rng)
    assert not mdp.state_is_terminal(state)
    mdp.action_space.seed(123)

    states = [state]
    for _ in range(5):
        action = mdp.action_space.sample()
        next_state = mdp.sample_next_state(state, action, rng)
        states.append(state)
        rew = mdp.get_reward(state, action, next_state)
        assert np.isclose(rew, 0.0)
        state = next_state

    # Uncomment for visualization.
    import imageio.v2 as iio

    imgs = [mdp.render_state(s) for s in states]
    iio.mimsave("pybullet_handover_test.gif", imgs)
