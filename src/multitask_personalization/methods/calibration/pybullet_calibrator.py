"""A domain-specific parameter setting method for pybullet tasks."""

import numpy as np

from multitask_personalization.envs.intake_process import (
    IntakeAction,
    IntakeObservation,
)
from multitask_personalization.envs.pybullet.pybullet_sim import (
    PyBulletSimulator,
)
from multitask_personalization.envs.pybullet.pybullet_task_spec import (
    PyBulletTaskSpec,
)
from multitask_personalization.methods.calibration.calibrator import Calibrator
from multitask_personalization.methods.policies.parameterized_policy import (
    PolicyParameters,
)


class PyBulletCalibrator(Calibrator):
    """A domain-specific parameter setting method for pybullet tasks."""

    def __init__(self, task_spec: PyBulletTaskSpec) -> None:
        self._sim = PyBulletSimulator(task_spec)

    def get_parameters(
        self, task_id: str, intake_data: list[tuple[IntakeAction, IntakeObservation]]
    ) -> PolicyParameters:
        # Find decision boundary between maximal positive and minimal negative.
        # max_positive: float | None = None
        # min_negative: float | None = None
        # for action, obs in intake_data:
        #     # check real ROM model implementation.
        #     dist = np.sqrt(
        #         np.sum(np.subtract(action, self._sim.rom_sphere_center) ** 2)
        #     )
        #     if obs:
        #         if max_positive is None or dist > max_positive:
        #             max_positive = dist
        #     else:
        #         if min_negative is None or dist < min_negative:
        #             min_negative = dist
        # if max_positive is None or min_negative is None:
        #     return np.inf
        # params = (max_positive + min_negative) / 2

        lower_bound = [-0.2, -0.25, 0.15, 0.0]
        upper_bound = [0.1, -0.1, 0.5, 0.3]
        params = np.random.uniform(lower_bound, upper_bound)
        return params


class OraclePyBulletCalibrator(Calibrator):
    """A domain-specific calibrator for pybullet that uses oracle info."""

    def __init__(self, task_spec: PyBulletTaskSpec) -> None:
        self._sim = PyBulletSimulator(task_spec)

    def get_parameters(
        self, task_id: str, intake_data: list[tuple[IntakeAction, IntakeObservation]]
    ) -> PolicyParameters:
        # check real ROM model implementation.
        # return self._sim.rom_sphere_radius

        return self._sim.parameterized_rom_model.get_rom_model_context_parameters()
