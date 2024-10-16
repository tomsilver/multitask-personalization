"""The intake process for the pybullet environment."""

from __future__ import annotations

from functools import cached_property

import gymnasium as gym
import numpy as np
from tomsutils.spaces import EnumSpace

from multitask_personalization.envs.intake_process import IntakeProcess
from multitask_personalization.envs.pybullet.pybullet_sim import (
    PyBulletSimulator,
)
from multitask_personalization.envs.pybullet.pybullet_structs import (
    _PyBulletIntakeAction,
    _PyBulletIntakeObs,
)
from multitask_personalization.structs import (
    CategoricalDistribution,
)


class PyBulletIntakeProcess(IntakeProcess[_PyBulletIntakeObs, _PyBulletIntakeAction]):
    """Intake process for the pybullet handover environment."""

    def __init__(self, horizon: int, sim: PyBulletSimulator) -> None:
        self._horizon = horizon
        self._sim = sim

    @cached_property
    def observation_space(self) -> EnumSpace[_PyBulletIntakeObs]:
        return EnumSpace([True, False])

    @cached_property
    def action_space(self) -> gym.spaces.Box:
        x, y, z = self._sim.rom_sphere_center
        size = 0.5
        return gym.spaces.Box(
            low=np.array([x - size, y - size, z - size], dtype=np.float32),
            high=np.array([x + size, y + size, z + size], dtype=np.float32),
        )

    @property
    def horizon(self) -> int:
        return self._horizon

    def get_observation_distribution(
        self,
        action: _PyBulletIntakeAction,
    ) -> CategoricalDistribution[_PyBulletIntakeObs]:
        dist = np.sqrt(np.sum(np.subtract(action, self._sim.rom_sphere_center) ** 2))
        result = dist < self._sim.rom_sphere_radius
        return CategoricalDistribution({result: 1.0, not result: 0.0})
