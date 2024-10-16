"""Tests for pybullet_handover_policy.py."""

import numpy as np

from multitask_personalization.envs.pybullet.pybullet_handover import (
    PyBulletHandoverTask,
)
from multitask_personalization.methods.policies.pybullet_handover_policy import (
    PyBulletHandoverParameterizedPolicy,
)


def test_pybullet_handover_policy():
    """Tests for pybullet_handover_policy.py."""
    task = PyBulletHandoverTask(
        intake_horizon=5,
        use_gui=False,
    )
    mdp = task.mdp
    rng = np.random.default_rng(123)
    state = mdp.sample_initial_state(rng)
    assert not mdp.state_is_terminal(state)
    mdp.action_space.seed(123)

    policy = PyBulletHandoverParameterizedPolicy(task.scene_description)
    params = 0.2  # radius of ROM sphere
    policy.reset(task.id, params)

    states = [state]
    for _ in range(500):
        action = policy.step(state)
        next_state = mdp.sample_next_state(state, action, rng)
        states.append(state)
        rew = mdp.get_reward(state, action, next_state)
        if rew > 0:
            break
        state = next_state

    # Uncomment for visualization.
    # import imageio.v2 as iio
    # imgs = [mdp.render_state(s) for s in states]
    # iio.mimsave("pybullet_handover_policy_test.mp4", imgs)
