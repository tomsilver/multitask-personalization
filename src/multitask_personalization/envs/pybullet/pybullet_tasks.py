"""Tasks for the pybullet environment."""

from __future__ import annotations

import pybullet as p

from multitask_personalization.envs.pybullet.pybullet_intake_process import (
    PyBulletHandoverIntakeProcess,
)
from multitask_personalization.envs.pybullet.pybullet_mdp import PyBulletHandoverMDP
from multitask_personalization.envs.pybullet.pybullet_scene_description import (
    PyBulletHandoverSceneDescription,
)
from multitask_personalization.envs.pybullet.pybullet_sim import (
    PyBulletHandoverSimulator,
)
from multitask_personalization.envs.task import Task


class PyBulletHandoverTask(Task):
    """The full handover task."""

    def __init__(
        self,
        intake_horizon: int,
        scene_description: PyBulletHandoverSceneDescription | None = None,
        use_gui: bool = False,
    ) -> None:

        self._intake_horizon = intake_horizon
        self._use_gui = use_gui

        # Finalize the scene description.
        if scene_description is None:
            scene_description = PyBulletHandoverSceneDescription()
        self.scene_description = scene_description

        # Generate a shared PyBullet simulator.
        self._sim = PyBulletHandoverSimulator(scene_description, use_gui)

    @property
    def id(self) -> str:
        return "handover"

    @property
    def mdp(self) -> PyBulletHandoverMDP:
        return PyBulletHandoverMDP(self._sim)

    @property
    def intake_process(self) -> PyBulletHandoverIntakeProcess:
        return PyBulletHandoverIntakeProcess(self._intake_horizon, self._sim)

    def close(self) -> None:
        p.disconnect(self._sim.physics_client_id)
