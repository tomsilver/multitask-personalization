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


@dataclass(frozen=True)
class CSPConstraint:
    """Constraint satisfaction problem constraint."""

    name: str
    variables: list[CSPVariable]
    constraint_fn: Callable[..., bool]  # inputs are CSPVariable values

    def check_solution(self, sol: dict[CSPVariable, Any]) -> bool:
        """Check whether the constraint holds given values of the varaibles."""
        vals = [sol[v] for v in self.variables]
        return self.constraint_fn(*vals)


@dataclass(frozen=True)
class CSP:
    """Constraint satisfaction problem."""

    variables: list[CSPVariable]
    constraints: list[CSPConstraint]

    def check_solution(self, sol: dict[CSPVariable, Any]) -> bool:
        """Check whether all constraints hold given values of the varaibles."""
        for constraint in self.constraints:
            if not constraint.check_solution(sol):
                return False
        return True


class CSPSampler(abc.ABC):
    """Samples values of one or more variables in a CSP conditioned.

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

    def reset(self, solution: dict[CSPVariable, Any]) -> None:
        """Reset the policy given a solution to the CSP."""
        self._current_solution = solution

    @abc.abstractmethod
    def step(self, obs: ObsType) -> ActType:
        """Return an action and advance any memory assuming action executes."""
