"""Tasks for the pybullet environment."""

from __future__ import annotations

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose

from multitask_personalization.envs.pybullet.pybullet_intake_process import (
    PyBulletIntakeProcess,
)
from multitask_personalization.envs.pybullet.pybullet_mdp import PyBulletMDP
from multitask_personalization.envs.pybullet.pybullet_sim import (
    PyBulletSimulator,
)
from multitask_personalization.envs.pybullet.pybullet_task_spec import (
    PyBulletTaskSpec,
)
from multitask_personalization.envs.task import Task


class PyBulletTask(Task):
    """A full pybullet task."""

    def __init__(
        self,
        intake_horizon: int,
        task_spec: PyBulletTaskSpec | None = None,
        use_gui: bool = False,
    ) -> None:

        self._intake_horizon = intake_horizon
        self._use_gui = use_gui

        # Finalize the scene description.
        if task_spec is None:
            task_spec = PyBulletTaskSpec()
        self.task_spec = task_spec

        # Generate a shared PyBullet simulator.
        self._sim = PyBulletSimulator(task_spec, use_gui)

    @property
    def id(self) -> str:
        return self.task_spec.task_name

    @property
    def mdp(self) -> PyBulletMDP:
        return PyBulletMDP(self._sim)

    @property
    def intake_process(self) -> PyBulletIntakeProcess:
        return PyBulletIntakeProcess(self._intake_horizon, self._sim)

    def close(self) -> None:
        p.disconnect(self._sim.physics_client_id)


def sample_pybullet_task_spec(rng: np.random.Generator) -> PyBulletTaskSpec:
    """Sample a task specification."""
    # Create a default task spec so we can generate values relative to default.
    default_spec = PyBulletTaskSpec()

    # Sample a handover task.
    task_name = "handover"

    # Randomize the initial position of the object.
    table_pose = default_spec.table_pose
    table_half_extents = default_spec.table_half_extents
    object_radius = default_spec.object_radius
    object_length = default_spec.object_length
    lb = (
        table_pose.position[0] - table_half_extents[0] + object_radius,
        table_pose.position[1] - table_half_extents[1] + object_radius,
        table_pose.position[2] + table_half_extents[2] + object_length / 2,
    )
    ub = (
        table_pose.position[0] + table_half_extents[0] - object_radius,
        table_pose.position[1] + table_half_extents[1] - object_radius,
        table_pose.position[2] + table_half_extents[2] + object_length / 2,
    )
    position = rng.uniform(lb, ub)
    object_pose = Pose(tuple(position))

    return PyBulletTaskSpec(task_name=task_name, object_pose=object_pose)
