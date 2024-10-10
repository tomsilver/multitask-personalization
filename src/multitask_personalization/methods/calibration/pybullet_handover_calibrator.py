"""A domain-specific parameter setting method for pybullet handover tasks."""

import numpy as np

from multitask_personalization.envs.intake_process import (
    IntakeAction,
    IntakeObservation,
)
from multitask_personalization.envs.pybullet_handover import (
    PyBulletHandoverSceneDescription,
    PyBulletHandoverSimulator,
)
from multitask_personalization.methods.calibration.calibrator import Calibrator
from multitask_personalization.methods.policies.parameterized_policy import (
    PolicyParameters,
)


class PyBulletHandoverCalibrator(Calibrator):
    """A domain-specific parameter setting method for pybullet handover
    tasks."""

    def __init__(self, scene_description: PyBulletHandoverSceneDescription) -> None:
        self._sim = PyBulletHandoverSimulator(scene_description)

    def get_parameters(
        self, task_id: str, intake_data: list[tuple[IntakeAction, IntakeObservation]]
    ) -> PolicyParameters:
        # Find decision boundary between maximal positive and minimal negative.
        max_positive: float | None = None
        min_negative: float | None = None
        for action, obs in intake_data:
            dist = np.sqrt(
                np.sum(np.subtract(action, self._sim.rom_sphere_center) ** 2)
            )
            if obs:
                if max_positive is None or dist > max_positive:
                    max_positive = dist
            else:
                if min_negative is None or dist < min_negative:
                    min_negative = dist
        if max_positive is None or min_negative is None:
            return np.inf
        params = (max_positive + min_negative) / 2
        return params


class OraclePyBulletHandoverCalibrator(Calibrator):
    """A domain-specific calibrator for pybullet handover that uses oracle
    info."""

    def __init__(self, scene_description: PyBulletHandoverSceneDescription) -> None:
        self._sim = PyBulletHandoverSimulator(scene_description)

    def get_parameters(
        self, task_id: str, intake_data: list[tuple[IntakeAction, IntakeObservation]]
    ) -> PolicyParameters:
        return self._sim.rom_sphere_radius
