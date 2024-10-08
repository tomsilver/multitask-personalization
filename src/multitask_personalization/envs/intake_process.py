"""A generic definition of a finite-horizon intake process."""

import abc
from typing import Any, Generic, TypeAlias, TypeVar

import gymnasium as gym
import numpy as np

from multitask_personalization.structs import (
    CategoricalDistribution,
)

IntakeObservation: TypeAlias = Any
IntakeAction: TypeAlias = Any


_O = TypeVar("_O", bound=IntakeObservation)
_U = TypeVar("_U", bound=IntakeAction)


class IntakeProcess(Generic[_O, _U]):
    """A finite-horizon intake process."""

    @property
    @abc.abstractmethod
    def observation_space(self) -> gym.Space:
        """Representation of the observation space."""

    @property
    @abc.abstractmethod
    def action_space(self) -> gym.Space:
        """Representation of the action space."""

    @property
    @abc.abstractmethod
    def horizon(self) -> int:
        """Length of the intake process."""

    @abc.abstractmethod
    def get_observation_distribution(
        self,
        action: _U,
    ) -> CategoricalDistribution[_O]:
        """Return a discrete distribution over observations."""

    def sample_next_observation(self, action: _U, rng: np.random.Generator) -> _O:
        """Sample an observation from the distribution.

        This function may be overwritten by subclasses when the explicit
        distribution is too large to enumerate.
        """
        return self.get_observation_distribution(action).sample(rng)
