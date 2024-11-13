"""Common data structures."""

import abc
from dataclasses import dataclass
from typing import Any, Callable, Generic

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType


@dataclass(frozen=True)
class CSPVariable:
    """Constraint satisfaction problem variable."""

    name: str
    domain: gym.spaces.Space

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, CSPVariable)
        return self.name == other.name


class CSPConstraint(abc.ABC):
    """Constraint satisfaction problem constraint."""

    def __init__(self, name: str, variables: list[CSPVariable]):
        self.name = name
        self.variables = variables

    @abc.abstractmethod
    def check_solution(self, sol: dict[CSPVariable, Any]) -> bool:
        """Check whether the constraint holds given values of the variables."""


class FunctionalCSPConstraint(CSPConstraint):
    """A constraint defined by a function that outputs bools."""

    def __init__(
        self,
        name: str,
        variables: list[CSPVariable],
        constraint_fn: Callable[..., bool],
    ):
        super().__init__(name, variables)
        self.constraint_fn = constraint_fn

    def check_solution(self, sol: dict[CSPVariable, Any]) -> bool:
        vals = [sol[v] for v in self.variables]
        return self.constraint_fn(*vals)


class LogProbCSPConstraint(CSPConstraint):
    """A constraint defined by a function that outputs log probabilities.

    The constraint_logprob_fn is a function mapping variable assignments
    to a log probability that the constraint holds. The constraint is
    defined by this value being greater than a threshold.
    """

    def __init__(
        self,
        name: str,
        variables: list[CSPVariable],
        constraint_logprob_fn: Callable[..., float],
        threshold: float = np.log(0.95),
    ):
        super().__init__(name, variables)
        self.constraint_logprob_fn = constraint_logprob_fn
        self.threshold = threshold

    def check_solution(self, sol: dict[CSPVariable, Any]) -> bool:
        return self.get_logprob(sol) >= self.threshold

    def get_logprob(self, sol: dict[CSPVariable, Any]) -> float:
        """Get the log probability of the constraint holding."""
        vals = [sol[v] for v in self.variables]
        return self.constraint_logprob_fn(*vals)


@dataclass(frozen=True)
class CSPCost:
    """A cost function to be minimized over certain CSP variables."""

    name: str
    variables: list[CSPVariable]
    cost_fn: Callable[..., float]  # inputs are CSPVariable values

    def get_cost(self, sol: dict[CSPVariable, Any]) -> float:
        """Evaluate the cost function."""
        vals = [sol[v] for v in self.variables]
        return self.cost_fn(*vals)


@dataclass(frozen=True)
class CSP:
    """Constraint satisfaction problem."""

    variables: list[CSPVariable]
    constraints: list[CSPConstraint]
    cost: CSPCost | None = None

    def check_solution(self, sol: dict[CSPVariable, Any]) -> bool:
        """Check whether all constraints hold given values of the variables."""
        for constraint in self.constraints:
            if not constraint.check_solution(sol):
                return False
        return True

    def get_cost(self, sol: dict[CSPVariable, Any]) -> float:
        """Evaluate the cost function."""
        assert self.cost is not None
        return self.cost.get_cost(sol)


class CSPSampler(abc.ABC):
    """Samples values of one or more variables in a CSP.

    The sampler can optionally use existing bindings of variables, e.g.,
    for conditional sampling, or for MCMC-style sampling.
    """

    def __init__(self, csp: CSP, sampled_vars: set[CSPVariable]) -> None:
        assert sampled_vars.issubset(csp.variables)
        self._csp = csp
        self._sampled_vars = sampled_vars

    @abc.abstractmethod
    def sample(
        self, current_vals: dict[CSPVariable, Any], rng: np.random.Generator
    ) -> dict[CSPVariable, Any]:
        """Sample values for self.sampled_vars given values of all CSP vars."""


class FunctionalCSPSampler(CSPSampler):
    """A CSPSampler implemented with a function."""

    def __init__(
        self,
        fn: Callable[
            [dict[CSPVariable, Any], np.random.Generator], dict[CSPVariable, Any]
        ],
        csp: CSP,
        sampled_vars: set[CSPVariable],
    ) -> None:
        self._fn = fn
        super().__init__(csp, sampled_vars)

    def sample(
        self, current_vals: dict[CSPVariable, Any], rng: np.random.Generator
    ) -> dict[CSPVariable, Any]:
        sample = self._fn(current_vals, rng)
        # Validate.
        for v, val in sample.items():
            assert v in self._sampled_vars, f"Sampled {v}"
            assert v.domain.contains(val), f"Value {val} not in domain {v.domain}"
        return sample


class CSPPolicy(abc.ABC, Generic[ObsType, ActType]):
    """Implements a policy that is conditioned on the solution to a CSP."""

    def __init__(self, csp: CSP, seed: int = 0) -> None:
        self._csp = csp
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._current_solution: dict[CSPVariable, Any] | None = None
        self._csp_var_name_to_var = {v.name: v for v in self._csp.variables}

    def _get_value(self, var_name: str) -> Any:
        assert self._current_solution is not None
        return self._current_solution[self._csp_var_name_to_var[var_name]]

    def reset(self, solution: dict[CSPVariable, Any]) -> None:
        """Reset the policy given a solution to the CSP."""
        self._current_solution = solution

    @abc.abstractmethod
    def step(self, obs: ObsType) -> ActType:
        """Return an action.

        Note that the policy may have internal memory; this advances it.
        """

    @abc.abstractmethod
    def check_termination(self, obs: ObsType) -> bool:
        """Check if the policy should terminate given a "next" observation."""


class CSPGenerator(abc.ABC, Generic[ObsType, ActType]):
    """Generates CSPs, samplers, policies, and initializations; and learns from
    environment transitions."""

    def __init__(
        self,
        seed: int = 0,
        explore_epsilon: float = 0.5,
    ) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._explore_epsilon = explore_epsilon

    def generate(
        self,
        obs: ObsType,
        user_allows_explore: bool = False,
    ) -> tuple[
        CSP, list[CSPSampler], CSPPolicy[ObsType, ActType], dict[CSPVariable, Any]
    ]:
        """Generate a CSP, samplers, policy, and initialization.

        If user_allows_explore is False, then the generator should
        "exploit", taking the best actions possible under its current
        models. Otherwise it is free to "explore".
        """
        do_explore = user_allows_explore and self._rng.uniform() < self._explore_epsilon
        return self._generate(obs, do_explore=do_explore)

    @abc.abstractmethod
    def _generate(
        self,
        obs: ObsType,
        do_explore: bool = False,
    ) -> tuple[
        CSP, list[CSPSampler], CSPPolicy[ObsType, ActType], dict[CSPVariable, Any]
    ]:
        """The actual generation method that subclasses should implement."""

    @abc.abstractmethod
    def learn_from_transition(
        self,
        obs: ObsType,
        act: ActType,
        next_obs: ObsType,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Update the generator given the new data point."""

    def get_metrics(self) -> dict[str, float]:
        """Report any metrics, e.g., about learned constraint parameters."""
        return {}


class CSPConstraintGenerator(abc.ABC, Generic[ObsType, ActType]):
    """Generates constraints for a CSP and learns over time."""

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @abc.abstractmethod
    def generate(
        self,
        obs: ObsType,
        csp_vars: list[CSPVariable],
        constraint_name: str,
    ) -> CSPConstraint:
        """Generate a constraint."""

    @abc.abstractmethod
    def learn_from_transition(
        self,
        obs: ObsType,
        act: ActType,
        next_obs: ObsType,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Update the generator given the new data point."""

    def get_metrics(self) -> dict[str, float]:
        """Report any metrics, e.g., about learned constraint parameters."""
        return {}
