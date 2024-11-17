"""Different methods for solving CSPs."""

import abc
import itertools
import math
from typing import Any, Iterator

import numpy as np
from tomsutils.spaces import EnumSpace
from tqdm import tqdm

from multitask_personalization.structs import (
    CSP,
    CSPConstraint,
    CSPCost,
    CSPSampler,
    CSPVariable,
    DiscreteCSP,
    FunctionalCSPConstraint,
)


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

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

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


class BruteForceDiscreteCSPSolver(DiscreteCSPSolver, IterativeSolver):
    """A brute-force discrete CSP solver."""

    def __init__(
        self,
        seed: int,
        min_num_satisfying_solutions: int = 50,
        show_progress_bar: bool = True,
    ) -> None:
        super().__init__(seed)
        self._min_num_satisfying_solutions = min_num_satisfying_solutions
        self._show_progress_bar = show_progress_bar
        self._candidate_generator: Iterator[dict[CSPVariable, Any]] = iter([])

    def _reset_iterative_solver(self, csp: CSP) -> None:
        assert isinstance(csp, DiscreteCSP)
        variables = csp.variables
        values = [csp.get_domain_values(v) for v in variables]
        # Randomly shuffle values so that the CSP produces different solutions
        # if called multiple times (e.g. as part of incremental approach).
        for v in values:
            self._rng.shuffle(v)
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
            min_num_satisfying_solutions=self._min_num_satisfying_solutions,
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


class IncrementalCSPSolver(CSPSolver):
    """Solver family inspired by the "incremental" version of PDDLStream."""

    def __init__(
        self,
        seed: int,
        discrete_csp_solver: DiscreteCSPSolver,
        max_generations: int,
    ) -> None:
        super().__init__(seed)
        self._discrete_csp_solver = discrete_csp_solver
        self._max_generations = max_generations

    @abc.abstractmethod
    def _generate_candidates(
        self,
        csp: CSP,
        initialization: dict[CSPVariable, Any],
        samplers: list[CSPSampler],
        generation: int,
    ) -> dict[CSPVariable, list[Any]]:
        """Generate candidate values for each CSP variable."""

    def solve(
        self,
        csp: CSP,
        initialization: dict[CSPVariable, Any],
        samplers: list[CSPSampler],
    ) -> dict[CSPVariable, Any] | None:
        for generation in range(self._max_generations):
            candidates = self._generate_candidates(
                csp, initialization, samplers, generation
            )
            # Convert into a discrete CSP, first without costs, to check if
            # there is any feasible solution at all.
            discrete_csp = self._discretize_csp(csp, candidates, include_costs=False)
            sol = self._discrete_csp_solver.solve(discrete_csp)
            # If no solution is found, we need to keep expanding the candidates.
            if sol is None:
                continue
            # Otherwise, we found a solution, so we'll prepare to terminate.
            # Rerun discrete CSP optimization, but this time WITH costs.
            discrete_csp = self._discretize_csp(csp, candidates, include_costs=True)
            improved_sol = self._discrete_csp_solver.solve(discrete_csp)
            assert improved_sol is not None
            # Note that this solution may still not be optimal due to the fact
            # that we still only have finite candidates in the discrete CSP.
            assert csp.check_solution(improved_sol)
            return improved_sol
        # Failed to find a solution.
        return None

    def _discretize_csp(
        self, csp: CSP, candidates: dict[CSPVariable, list[Any]], include_costs: bool
    ) -> DiscreteCSP:
        """Convert into a discrete CSP given candidate variable values."""

        variable_to_discrete: dict[CSPVariable, CSPVariable] = {}
        for variable in csp.variables:
            discrete_variable = CSPVariable(
                variable.name, EnumSpace(candidates[variable])
            )
            variable_to_discrete[variable] = discrete_variable

        discrete_constraints: list[CSPConstraint] = []
        for constraint in csp.constraints:
            discrete_constraint = self._discretize_constraint(
                constraint, variable_to_discrete
            )
            discrete_constraints.append(discrete_constraint)

        if include_costs and csp.cost is not None:
            discrete_cost = CSPCost(
                csp.cost.name,
                [variable_to_discrete[v] for v in csp.cost.variables],
                csp.cost.cost_fn,
            )
        else:
            discrete_cost = None

        discrete_variables = list(variable_to_discrete.values())
        return DiscreteCSP(discrete_variables, discrete_constraints, discrete_cost)

    def _discretize_constraint(
        self,
        constraint: CSPConstraint,
        variable_to_discrete: dict[CSPVariable, CSPVariable],
    ) -> CSPConstraint:
        variables = [variable_to_discrete[v] for v in constraint.variables]
        constraint_fn = lambda *x: constraint.check_solution(dict(zip(variables, x)))
        return FunctionalCSPConstraint(constraint.name, variables, constraint_fn)


class TreeSearchIncrementalCSPSolver(IncrementalCSPSolver):
    """Generate candidates by tree search with progressive widening."""

    def __init__(
        self,
        seed: int,
        discrete_csp_solver: DiscreteCSPSolver,
        max_depth: int = 5,
        base_branching_factor: int = 5,
        progressive_widening_scale: float = 1.1,
    ) -> None:
        super().__init__(seed, discrete_csp_solver, max_generations=max_depth)
        self._discrete_csp_solver = discrete_csp_solver
        self._base_branching_factor = base_branching_factor
        self._progressive_widening_scale = progressive_widening_scale

    def _generate_candidates(
        self,
        csp: CSP,
        initialization: dict[CSPVariable, Any],
        samplers: list[CSPSampler],
        generation: int,
    ) -> dict[CSPVariable, list[Any]]:

        max_depth = generation
        branching_factor = int(
            self._base_branching_factor * (self._progressive_widening_scale**max_depth)
        )

        var_to_vals = {v: [val] for v, val in initialization.items()}

        queue = [(0, initialization.copy())]
        while queue:
            depth, sol = queue.pop()
            if depth >= max_depth:
                continue
            for sampler in samplers:
                for _ in range(branching_factor):
                    partial_sol = sampler.sample(sol, self._rng)
                    for variable, value in partial_sol.items():
                        var_to_vals[variable].append(value)
                    new_sol = sol.copy()
                    new_sol.update(partial_sol)
                    if depth < max_depth:
                        queue.append((depth + 1, new_sol))

        # Collapse unique values where possible.
        final_var_to_vals: dict[CSPVariable, list[Any]] = {}
        for v, vals in var_to_vals.items():
            try:
                vals = sorted(set(vals))
            except TypeError:
                pass
            final_var_to_vals[v] = vals

        return final_var_to_vals
