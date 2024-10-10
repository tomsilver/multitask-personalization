"""Base task, including an MDP and an intake process."""

import abc

from multitask_personalization.envs.intake_process import IntakeProcess
from multitask_personalization.envs.mdp import MDP


class Task(abc.ABC):
    """Base task, including an MDP and an intake process."""

    @property
    @abc.abstractmethod
    def id(self) -> str:
        """The identity of this task."""

    @property
    @abc.abstractmethod
    def mdp(self) -> MDP:
        """The MDP."""

    @property
    @abc.abstractmethod
    def intake_process(self) -> IntakeProcess:
        """The intake process."""

    @abc.abstractmethod
    def close(self) -> None:
        """Run at the end of experiments."""
