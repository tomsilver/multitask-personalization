"""CSP generators."""

import abc
from typing import Any, Generic

import numpy as np
from gymnasium.core import ActType, ObsType

from multitask_personalization.structs import (
    CSP,
    CSPConstraint,
    CSPCost,
    CSPPolicy,
    CSPSampler,
    CSPVariable,
    LogProbCSPConstraint,
)
from multitask_personalization.utils import bernoulli_entropy


class CSPGenerator(abc.ABC, Generic[ObsType, ActType]):
    """Generates CSPs, samplers, policies, and initializations; and learns from
    environment transitions."""

    def __init__(
        self,
        seed: int = 0,
        explore_method: str = "nothing-personal",
    ) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._explore_method = explore_method

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
        variables, initialization = self._generate_variables(obs)
        constraints = self._generate_constraints(obs, variables, user_allows_explore)
        cost = self._generate_cost(obs, variables, user_allows_explore)
        csp = CSP(variables, constraints, cost)
        samplers = self._generate_samplers(obs, csp)
        policy = self._generate_policy(obs, csp)
        return csp, samplers, policy, initialization

    @abc.abstractmethod
    def _generate_variables(
        self,
        obs: ObsType,
    ) -> tuple[list[CSPVariable], dict[CSPVariable, Any]]:
        """Generate CSP variables and an initialization for them."""

    def _generate_constraints(
        self,
        obs: ObsType,
        variables: list[CSPVariable],
        user_allows_explore: bool = False,
    ) -> list[CSPConstraint]:
        """Generate CSP constraints."""
        nonpersonal_constraints = self._generate_nonpersonal_constraints(obs, variables)
        if user_allows_explore and self._explore_method in (
            "nothing-personal",
            "max-entropy",
        ):
            return nonpersonal_constraints
        personal_constraints = self._generate_personal_constraints(obs, variables)
        return nonpersonal_constraints + personal_constraints

    @abc.abstractmethod
    def _generate_personal_constraints(
        self,
        obs: ObsType,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:
        """Generate personal CSP constraints."""

    @abc.abstractmethod
    def _generate_nonpersonal_constraints(
        self,
        obs: ObsType,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:
        """Generate nonpersonal CSP constraints."""

    def _generate_cost(
        self,
        obs: ObsType,
        variables: list[CSPVariable],
        user_allows_explore: bool = False,
    ) -> CSPCost | None:
        """Generate CSP costs."""
        if user_allows_explore and self._explore_method == "max-entropy":
            personal_lp_constraints = [
                c
                for c in self._generate_personal_constraints(obs, variables)
                if isinstance(c, LogProbCSPConstraint)
            ]
            num_personal_lp_constraints = len(personal_lp_constraints)
            if num_personal_lp_constraints == 0:
                return None

            def _max_entropy_fn(*args) -> float:
                total_entropy = 0.0
                sol = dict(zip(variables, args))
                for constraint in personal_lp_constraints:
                    lp = constraint.get_logprob(sol)
                    entropy = bernoulli_entropy(lp)
                    total_entropy += entropy
                mean_entropy = total_entropy / num_personal_lp_constraints
                return 1.0 - mean_entropy

            return CSPCost("maximize-entropy", variables, _max_entropy_fn)
        return self._generate_exploit_cost(obs, variables)

    @abc.abstractmethod
    def _generate_exploit_cost(
        self,
        obs: ObsType,
        variables: list[CSPVariable],
    ) -> CSPCost | None:
        """Generate CSPs costs assuming that we're not exploring."""

    @abc.abstractmethod
    def _generate_samplers(
        self,
        obs: ObsType,
        csp: CSP,
    ) -> list[CSPSampler]:
        """Generate samplers for CSP variables."""

    @abc.abstractmethod
    def _generate_policy(
        self,
        obs: ObsType,
        csp: CSP,
    ) -> CSPPolicy:
        """Generate a policy conditioned on a CSP solution."""

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
