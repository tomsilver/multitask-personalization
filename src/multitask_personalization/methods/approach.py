"""Base approach definition."""

import abc
from pathlib import Path

import gymnasium as gym
from tomsutils.gym_agent import Agent, _ActType, _ObsType


class BaseApproach(Agent[_ObsType, _ActType]):
    """A generic base approach for gym environments."""

    def __init__(self, action_space: gym.spaces.Space[_ActType], seed: int):
        super().__init__(seed)
        self._action_space = action_space
        self._seed = seed

    @abc.abstractmethod
    def _get_action(self) -> _ActType:
        """The main action selection method."""

    def get_step_metrics(self) -> dict[str, float]:
        """Return any approach-specific metrics for the present episode."""
        return {}

    @abc.abstractmethod
    def save(self, model_dir: Path) -> None:
        """Save sufficient information for fully loading the model."""

    @abc.abstractmethod
    def load(self, model_dir: Path) -> None:
        """Load from a saved model directory."""


class ApproachFailure(Exception):
    """Raised when an approach fails."""
