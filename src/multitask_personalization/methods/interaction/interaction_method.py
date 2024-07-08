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
        action_space: set[_U],
        observation_space: set[_O],
        seed: int,
    ) -> None:
        self._action_space = action_space
        self._observation_space = observation_space
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @abc.abstractmethod
    def get_action(self) -> _U:
        """Get a next action to execute."""

    @abc.abstractmethod
    def observe(self, obs: _O) -> None:
        """Record an observation."""
