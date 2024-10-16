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
        scene_description: PyBulletTaskSpec | None = None,
        use_gui: bool = False,
    ) -> None:

        self._intake_horizon = intake_horizon
        self._use_gui = use_gui

        # Finalize the scene description.
        if scene_description is None:
            scene_description = PyBulletTaskSpec()
        self.scene_description = scene_description

        # Generate a shared PyBullet simulator.
        self._sim = PyBulletSimulator(scene_description, use_gui)

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
