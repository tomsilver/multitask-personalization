"""Base class for intake interaction methods."""

import abc
from typing import Generic, TypeVar

import numpy as np

from multitask_personalization.envs.intake_process import (
    IntakeAction,
    IntakeObservation,
)

_U = TypeVar("_U", bound=IntakeAction)
_O = TypeVar("_O", bound=IntakeObservation)


class InteractionMethod(Generic[_U, _O]):
    """Base class for intake interaction methods."""

    def __init__(
        self,
        seed: int,
    ) -> None:
        self._current_task_id: str | None = None
        self._current_action_space: set[_U] | None = None
        self._current_observation_space: set[_O] | None = None
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(
        self, task_id: str, action_space: set[_U], observation_space: set[_O]
    ) -> None:
        """Called on task reset."""
        self._current_task_id = task_id
        self._current_action_space = action_space
        self._current_observation_space = observation_space

    @abc.abstractmethod
    def get_action(self) -> _U:
        """Get a next action to execute."""

    @abc.abstractmethod
    def observe(self, obs: _O) -> None:
        """Record an observation."""
