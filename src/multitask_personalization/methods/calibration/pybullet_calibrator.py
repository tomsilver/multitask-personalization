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
        # placeholder: return random parameters
        lower_bound = [-0.3, -0.3, 0.0, 0.0]
        upper_bound = [0.3, 0.0, 0.5, 0.5]
        params = np.random.uniform(lower_bound, upper_bound)
        return params


class OraclePyBulletCalibrator(Calibrator):
    """A domain-specific calibrator for pybullet that uses oracle info."""

    def __init__(self, task_spec: PyBulletTaskSpec) -> None:
        self._sim = PyBulletSimulator(task_spec)

    def get_parameters(
        self, task_id: str, intake_data: list[tuple[IntakeAction, IntakeObservation]]
    ) -> PolicyParameters:
        # directly return the context parameters from the ROM model
        return self._sim.parameterized_rom_model.get_rom_model_context_parameters()
