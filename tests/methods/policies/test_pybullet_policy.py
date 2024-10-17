"""Tests for pybullet_policy.py."""

import numpy as np
from pybullet_helpers.geometry import Pose

from multitask_personalization.envs.pybullet.pybullet_task_spec import PyBulletTaskSpec
from multitask_personalization.envs.pybullet.pybullet_tasks import (
    PyBulletTask,
)
from multitask_personalization.methods.policies.pybullet_policy import (
    PyBulletParameterizedPolicy,
)


def test_pybullet_policy():
    """Tests for pybullet_policy.py."""
    rng = np.random.default_rng(123)
    task_specs = [
        # PyBulletTaskSpec(
        #     task_objective="hand over cup",
        #     tray_pose=Pose((-1000, -1000, -1000)),
        #     side_table_pose=Pose((-1000, -1000, -1000)),
        # ),
        # PyBulletTaskSpec(task_objective="hand over book"),
        PyBulletTaskSpec(task_objective="place books on tray"),
    ]
    for task_spec in task_specs:
        task = PyBulletTask(
            intake_horizon=5,
            task_spec=task_spec,
            use_gui=False,
        )
        mdp = task.mdp
        state = mdp.sample_initial_state(rng)
        assert not mdp.state_is_terminal(state)
        mdp.action_space.seed(123)

        policy = PyBulletParameterizedPolicy(task.task_spec)
        params = 0.3  # radius of ROM sphere
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
        # iio.mimsave("pybullet_policy_test.mp4", imgs)

        task.close()
