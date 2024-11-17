"""Different methods for solving CSPs."""

import abc
import itertools
import math
from typing import Any, Iterator

import numpy as np
from tqdm import tqdm

from multitask_personalization.structs import CSP, CSPSampler, CSPVariable, DiscreteCSP


class CSPSolver(abc.ABC):
    """A CSP solver."""

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @abc.abstractmethod
    def solve(
        self,
        csp: CSP,
        initialization: dict[CSPVariable, Any],
        samplers: list[CSPSampler],
    ) -> dict[CSPVariable, Any] | None:
        """Solve the given CSP."""


class DiscreteCSPSolver(abc.ABC):
    """A solver of DiscreteCSPs."""

    @abc.abstractmethod
    def solve(
        self,
        csp: DiscreteCSP,
    ) -> dict[CSPVariable, Any] | None:
        """Solve the given DiscreteCSP."""


class IterativeSolver(abc.ABC):
    """A mix-in for CSPSolver or DiscreteCSPSolver."""

    @abc.abstractmethod
    def _reset_iterative_solver(self, csp: CSP) -> None:
        """Reset any memory in the iterative solver."""

    @abc.abstractmethod
    def _get_next_iterative_candidate(
        self, sol: dict[CSPVariable, Any], sat: bool, cost: float
    ) -> dict[CSPVariable, Any] | None:
        """Get the next candidate to consider, or None to finish."""

    def _solve_iteratively(
        self,
        csp: CSP,
        initialization: dict[CSPVariable, Any],
        max_iters: int,
        min_num_satisfying_solutions: int,
        show_progress_bar: bool,
    ) -> dict[CSPVariable, Any] | None:
        self._reset_iterative_solver(csp)
        sol = initialization.copy()
        best_satisfying_sol: dict[CSPVariable, Any] | None = None
        best_satisfying_cost: float = np.inf
        num_satisfying_solutions = 0
        for _ in (pbar := tqdm(range(max_iters), disable=not show_progress_bar)):
            pbar.set_description(f"Found {num_satisfying_solutions} solns")
            sat = csp.check_solution(sol)
            sol_cost = np.nan
            if sat:
                num_satisfying_solutions += 1
                if csp.cost is None:
                    return sol
                sol_cost = csp.get_cost(sol)
                if sol_cost < best_satisfying_cost:
                    best_satisfying_cost = sol_cost
                    best_satisfying_sol = sol
                if num_satisfying_solutions >= min_num_satisfying_solutions:
                    return best_satisfying_sol
            candidate = self._get_next_iterative_candidate(sol, sat, sol_cost)
            if candidate is None:
                break
            sol = candidate
        return best_satisfying_sol


class ExhaustiveDiscreteCSPSolver(DiscreteCSPSolver, IterativeSolver):
    """A brute-force exhaustive discrete CSP solver."""

    def __init__(
        self,
        show_progress_bar: bool = True,
    ) -> None:
        self._show_progress_bar = show_progress_bar
        self._candidate_generator: Iterator[dict[CSPVariable, Any]] = iter([])

    def _reset_iterative_solver(self, csp: CSP) -> None:
        assert isinstance(csp, DiscreteCSP)
        variables = csp.variables
        values = [csp.get_domain_values(v) for v in variables]
        self._candidate_generator = (
            dict(zip(variables, vals)) for vals in itertools.product(*values)
        )

    def _get_next_iterative_candidate(
        self, sol: dict[CSPVariable, Any], sat: bool, cost: float
    ) -> dict[CSPVariable, Any] | None:
        try:
            return next(self._candidate_generator)
        except StopIteration:
            return None

    def solve(
        self,
        csp: DiscreteCSP,
    ) -> dict[CSPVariable, Any] | None:
        variables = csp.variables
        values = [csp.get_domain_values(v) for v in variables]
        num_values = [len(v) for v in values]
        initialization = {k: v[0] for k, v in zip(variables, values)}
        max_iters = math.prod(num_values)
        return self._solve_iteratively(
            csp,
            initialization,
            max_iters,
            min_num_satisfying_solutions=max_iters,
            show_progress_bar=self._show_progress_bar,
        )


class RandomWalkCSPSolver(CSPSolver, IterativeSolver):
    """Call samplers completely at random and remember the best seen
    solution."""

    def __init__(
        self,
        seed: int,
        max_iters: int = 100_000,
        min_num_satisfying_solutions: int = 50,
        show_progress_bar: bool = True,
    ) -> None:
        super().__init__(seed)
        self._max_iters = max_iters
        self._min_num_satisfying_solutions = min_num_satisfying_solutions
        self._show_progress_bar = show_progress_bar
        self._current_samplers: list[CSPSampler] = []

    def _reset_iterative_solver(self, csp: CSP) -> None:
        pass  # doesn't use any memory

    def _get_next_iterative_candidate(
        self, sol: dict[CSPVariable, Any], sat: bool, cost: float
    ) -> dict[CSPVariable, Any] | None:
        samplers = self._current_samplers
        sampler = samplers[self._rng.choice(len(samplers))]
        partial_sol = sampler.sample(sol, self._rng)
        sol = sol.copy()
        sol.update(partial_sol)
        return sol

    def solve(
        self,
        csp: CSP,
        initialization: dict[CSPVariable, Any],
        samplers: list[CSPSampler],
    ) -> dict[CSPVariable, Any] | None:
        self._current_samplers = samplers
        return self._solve_iteratively(
            csp,
            initialization,
            self._max_iters,
            self._min_num_satisfying_solutions,
            self._show_progress_bar,
        )


# class IncrementalCSPSolver(CSPSolver):
#     """Solver inspired by the "incremental" version of PDDLStream."""

#     def __init__(
#         self,
#         seed: int,
#         discrete_csp_solver: DiscreteCSPSolver,
#         max_depth: int = 5,
#         base_branching_factor: int = 5,
#         progressive_widening_scale: float = 1.1,
#     ) -> None:
#         super().__init__(seed)
#         self._discrete_csp_solver = discrete_csp_solver
#         self._base_branching_factor = base_branching_factor
#         self._max_depth = max_depth
#         self._progressive_widening_scale = progressive_widening_scale

#     def solve(
#         self,
#         csp: CSP,
#         initialization: dict[CSPVariable, Any],
#         samplers: list[CSPSampler],
#     ) -> dict[CSPVariable, Any] | None:

#         branching_factor = self._base_branching_factor
#         for depth in range(1, self._max_depth + 1):
#             # Generate candidates up to the given breadth and depth.
#             candidates = self._generate_candidates(initialization, samplers, branching_factor, depth)
#             # Convert into a discrete CSP and try to optimize.
#             discrete_csp = self._discretize_csp(csp, candidates)
#             sol = self._discrete_csp_solver.solve(discrete_csp)
#             # Terminate immediately if a solution is found. Note that this may
#             # be suboptimal.
#             if sol is not None:
#                 assert csp.check_solution(sol)
#                 return sol
#             # Widen branching factor.
#             branching_factor = branching_factor * self._progressive_widening_scale

#         # Failed to find a solution.
#         return None

#     def _generate_candidates(self, initialization: dict[CSPVariable, Any],
#         samplers: list[CSPSampler],
#         branching_factor: int,
#         depth: int) -> dict[CSPVariable, list[Any]]:
#         """Generate candidate values for each CSP variable."""
#         import ipdb; ipdb.set_trace()

#     def _discretize_csp(self, csp: CSP, candidates: dict[CSPVariable, list[Any]]) -> DiscreteCSP:
#         """Convert into a discrete CSP given candidate variable values."""
#         import ipdb; ipdb.set_trace()
