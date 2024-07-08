"""Base method for policy parameter setting."""

import abc
from typing import Generic, TypeVar

from multitask_personalization.envs.intake_process import (
    IntakeAction,
    IntakeObservation,
)
from multitask_personalization.policies.parameterized_policy import PolicyParameters

_U = TypeVar("_U", bound=IntakeAction)
_O = TypeVar("_O", bound=IntakeObservation)
_P = TypeVar("_P", bound=PolicyParameters)


class ParameterSettingMethod(Generic[_U, _O, _P]):
    """Base method for policy parameter setting."""

    @abc.abstractmethod
    def get_parameters(self, task_id: str, intake_data: list[tuple[_U, _O]]) -> _P:
        """Get new parameters for the given task with given intake data."""
