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
    default_task_spec = PyBulletTaskSpec()
    task_specs = [
        PyBulletTaskSpec(
            task_objective="hand over cup",
            tray_pose=Pose((-1000, -1000, -1000)),
            side_table_pose=Pose((-1000, -1000, -1000)),
        ),
        PyBulletTaskSpec(task_objective="hand over book"),
        PyBulletTaskSpec(
            task_objective="place books on tray",
            book_poses=default_task_spec.book_poses[:2],
            book_half_extents=default_task_spec.book_half_extents[:2],
            book_rgbas=default_task_spec.book_rgbas[:2],
        ),
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

        policy = PyBulletParameterizedPolicy(
            task.task_spec, max_motion_planning_time=0.1
        )
        # params = 0.3  # radius of ROM sphere
        params = np.array([0.0251, -0.2047, 0.3738, 0.1586])
        policy.reset(task.id, params)

        states = [state]
        for _ in range(10000):
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
