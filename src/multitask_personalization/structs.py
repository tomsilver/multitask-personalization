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

    def __init__(self, name: str, variables: list[CSPVariable]) -> None:
        self.name = name
        self.variables = variables

    @abc.abstractmethod
    def constraint_fn(self, *args) -> bool:
        """The arguments are values of self.variables."""

    def check_solution(self, sol: dict[CSPVariable, Any]) -> bool:
        """Check whether the constraint holds given values of the variables."""
        vals = [sol[v] for v in self.variables]
        return self.constraint_fn(*vals)


class FunctionalCSPConstraint(CSPConstraint):
    """A constraint defined by a given function."""

    def __init__(
        self,
        name: str,
        variables: list[CSPVariable],
        constraint_fn: Callable[..., bool],
    ) -> None:
        super().__init__(name, variables)
        self._constraint_fn = constraint_fn

    def constraint_fn(self, *args) -> bool:
        return self._constraint_fn(*args)


class LevelSetCSPConstraint(CSPConstraint):
    """Constraint defined as the zero-level-set of a score function.

    The score function should output values between 0 and 1.

    This type of constraint is useful for exploration.
    """

    def __init__(
        self,
        name: str,
        variables: list[CSPVariable],
        score_fn: Callable[..., float],
        padding: float = 1e-4,
    ) -> None:
        super().__init__(name, variables)
        self.score_fn = score_fn
        self._padding = padding  # for numerical stability

    def constraint_fn(self, *args) -> bool:
        score = self.score_fn(*args)
        assert 0.0 <= score <= 1.0
        return score < self._padding


@dataclass(frozen=True)
class CSP:
    """Constraint satisfaction problem."""

    variables: list[CSPVariable]
    constraints: list[CSPConstraint]

    def check_solution(self, sol: dict[CSPVariable, Any]) -> bool:
        """Check whether all constraints hold given values of the variables."""
        for constraint in self.constraints:
            if not constraint.check_solution(sol):
                return False
        return True


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
        """Return an action and advance any memory assuming action executes."""


class CSPGenerator(abc.ABC, Generic[ObsType, ActType]):
    """Generates CSPs, samplers, policies, and initializations; and learns from
    environment transitions."""

    def __init__(
        self,
        seed: int = 0,
        explore_method: str = "nothing-personal",
        ensemble_explore_threshold: float = 1e-1,
        ensemble_explore_members: int = 5,
        neighborhood_explore_max_radius: float = 1.0,
        neighborhood_explore_radius_decay: float = 0.99,
    ) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._explore_method = explore_method
        self._ensemble_explore_threshold = ensemble_explore_threshold
        self._ensemble_explore_members = ensemble_explore_members
        self._neighborhood_explore_max_radius = neighborhood_explore_max_radius
        self._neighborhood_explore_radius_decay = neighborhood_explore_radius_decay

    @abc.abstractmethod
    def generate(
        self,
        obs: ObsType,
        explore: bool = False,
    ) -> tuple[
        CSP, list[CSPSampler], CSPPolicy[ObsType, ActType], dict[CSPVariable, Any]
    ]:
        """Generate a CSP, samplers, policy, and initialization."""

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
        neighborhood: float = 0.0,
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


class EnsembleCSPConstraintGenerator(CSPConstraintGenerator[ObsType, ActType]):
    """A constraint generator implemented as an ensemble of constraint
    generators.

    This is useful for exploration. For example, to explore
    optimistically, you can use generate() with a small threshold.
    """

    def __init__(
        self, members: list[CSPConstraintGenerator[ObsType, ActType]], seed: int = 0
    ) -> None:
        super().__init__(seed)
        self._members = members

    def generate(
        self,
        obs: ObsType,
        csp_vars: list[CSPVariable],
        constraint_name: str,
        neighborhood: float = 0.0,
        member_classification_threshold: float = 0.5,
    ) -> CSPConstraint:

        member_constraints = [
            m.generate(obs, csp_vars, constraint_name, neighborhood=neighborhood)
            for m in self._members
        ]

        def _constraint_fn(*args) -> bool:
            total_pos = 0
            for constraint in member_constraints:
                total_pos += constraint.constraint_fn(*args)
            frac = total_pos / len(member_constraints)
            return frac >= member_classification_threshold

        constraint = FunctionalCSPConstraint(constraint_name, csp_vars, _constraint_fn)

        return constraint

    def learn_from_transition(
        self,
        obs: ObsType,
        act: ActType,
        next_obs: ObsType,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        for member in self._members:
            member.learn_from_transition(obs, act, next_obs, reward, done, info)

    def get_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for member_idx, member in enumerate(self._members):
            member_metrics = member.get_metrics()
            for metric_name, val in member_metrics.items():
                metrics[f"ensemble-{member_idx}-{metric_name}"] = val
        return metrics
