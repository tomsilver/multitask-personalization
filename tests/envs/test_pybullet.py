"""Tests for pybullet.py."""

import numpy as np

from multitask_personalization.envs.pybullet.pybullet_tasks import (
    PyBulletTask,
    sample_pybullet_task_spec,
)


def test_pybullet():
    """Tests for pybullet.py."""
    rng = np.random.default_rng(123)

    for _ in range(5):
        task_spec = sample_pybullet_task_spec(rng)
        task = PyBulletTask(
            intake_horizon=1,
            task_spec=task_spec,
            use_gui=False,
        )
        mdp = task.mdp
        state = mdp.sample_initial_state(rng)
        assert not mdp.state_is_terminal(state)
        mdp.action_space.seed(123)

        states = [state]
        for _ in range(10):
            action = mdp.action_space.sample()
            next_state = mdp.sample_next_state(state, action, rng)
            states.append(state)
            rew = mdp.get_reward(state, action, next_state)
            assert rew >= 0
            state = next_state

        # Uncomment for visualization.
        # import imageio.v2 as iio
        # imgs = [mdp.render_state(s) for s in states]
        # iio.mimsave(f"pybullet_test_{task_num}.gif", imgs)

        task.close()
