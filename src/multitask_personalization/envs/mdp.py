"""A generic definition of an MDP with discrete states and actions."""

import abc
from typing import Generic, TypeAlias, TypeVar

import numpy as np

from multitask_personalization.structs import (
    CategoricalDistribution,
    HashableComparable,
    Image,
)

MDPState: TypeAlias = HashableComparable
MDPAction: TypeAlias = HashableComparable


_S = TypeVar("_S", bound=MDPState)
_A = TypeVar("_A", bound=MDPAction)


class MDP(Generic[_S, _A]):
    """An indefinite-horizon Markov Decision Process."""

    @property
    @abc.abstractmethod
    def state_space(self) -> set[_S]:
        """Representation of the MDP state set."""

    @property
    @abc.abstractmethod
    def action_space(self) -> set[_A]:
        """Representation of the MDP action set."""

    @property
    def temporal_discount_factor(self) -> float:
        """Gamma, defaults to 1."""
        return 1.0

    @abc.abstractmethod
    def state_is_terminal(self, state: _S) -> bool:
        """Whether the state is terminal."""

    @abc.abstractmethod
    def get_reward(self, state: _S, action: _A, next_state: _S) -> float:
        """Return (deterministic) reward for executing action in state."""

    @abc.abstractmethod
    def get_initial_state_distribution(
        self,
    ) -> CategoricalDistribution[_S]:
        """Return a discrete distribution over initial states."""

    def sample_initial_state(self, rng: np.random.Generator) -> _S:
        """Sample an initial state from the distribution.

        This function may be overwritten by subclasses when the explicit
        distribution is too large to enumerate.
        """
        return self.get_initial_state_distribution().sample(rng)

    @abc.abstractmethod
    def get_transition_distribution(
        self, state: _S, action: _A
    ) -> CategoricalDistribution[_S]:
        """Return a discrete distribution over next states."""

    def sample_next_state(self, state: _S, action: _A, rng: np.random.Generator) -> _S:
        """Sample a next state from the transition distribution.

        This function may be overwritten by subclasses when the explicit
        distribution is too large to enumerate.
        """
        return self.get_transition_distribution(state, action).sample(rng)

    def get_transition_probability(
        self, state: _S, action: _A, next_state: _S
    ) -> float:
        """Convenience method for some algorithms."""
        return self.get_transition_distribution(state, action)(next_state)

    @abc.abstractmethod
    def render_state(self, state: _S) -> Image:
        """Optional rendering function for visualizations."""
