"""Base environment, including an MDP and an intake process."""

import abc

from multitask_personalization.envs.intake_process import IntakeProcess
from multitask_personalization.envs.mdp import MDP


class Env(abc.ABC):
    """Base environment, including an MDP and an intake process."""

    @property
    @abc.abstractmethod
    def mdp(self) -> MDP:
        """The MDP."""

    @property
    @abc.abstractmethod
    def intake_process(self) -> IntakeProcess:
        """The intake process."""
