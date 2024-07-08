"""Base class for parameterized policies."""

import abc
from typing import Any, TypeAlias, TypeVar, Generic
from multitask_personalization.envs.mdp import MDPState, MDPAction

PolicyParameters: TypeAlias = Any

_S = TypeVar("_S", bound=MDPState)
_A = TypeVar("_A", bound=MDPAction)
_P = TypeVar("_P", bound=PolicyParameters)


class ParameterizedPolicy(Generic[_S, _A, _P]):
    """Base class for parameterized policies."""

    def __init__(self) -> None:
        self._current_task_id: str | None = None
        self._current_parameters: _P | None = None

    def reset(self, task_id: str, parameters: _P) -> None:
        """Start a new task."""
        self._task_id = task_id
        self._current_parameters = parameters

    @abc.abstractmethod
    def step(self, state: _S) -> _A:
        """Get an action for the given state and advance time."""