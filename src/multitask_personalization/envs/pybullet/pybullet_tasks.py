"""Tasks for the pybullet environment."""

from __future__ import annotations

import pybullet as p

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
        return "pybullet"

    @property
    def mdp(self) -> PyBulletMDP:
        return PyBulletMDP(self._sim)

    @property
    def intake_process(self) -> PyBulletIntakeProcess:
        return PyBulletIntakeProcess(self._intake_horizon, self._sim)

    def close(self) -> None:
        p.disconnect(self._sim.physics_client_id)
