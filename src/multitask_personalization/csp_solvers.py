"""Different methods for solving CSPs."""

import abc
from typing import Any

import numpy as np
from tqdm import tqdm

from multitask_personalization.structs import CSP, CSPSampler, CSPVariable


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


class RandomWalkCSPSolver(CSPSolver):
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

    def solve(
        self,
        csp: CSP,
        initialization: dict[CSPVariable, Any],
        samplers: list[CSPSampler],
    ) -> dict[CSPVariable, Any] | None:
        sol = initialization.copy()
        best_satisfying_sol: dict[CSPVariable, Any] | None = None
        best_satisfying_cost: float = np.inf
        num_satisfying_solutions = 0
        sampler_idxs = list(range(len(samplers)))
        for _ in (
            pbar := tqdm(range(self._max_iters), disable=not self._show_progress_bar)
        ):
            pbar.set_description(f"Found {num_satisfying_solutions} solns")
            if csp.check_solution(sol):
                num_satisfying_solutions += 1
                if csp.cost is None:
                    # Uncomment to debug.
                    # from multitask_personalization.utils import print_csp_sol
                    # print_csp_sol(sol)
                    return sol
                cost = csp.get_cost(sol)
                if cost < best_satisfying_cost:
                    best_satisfying_cost = cost
                    best_satisfying_sol = sol
                if num_satisfying_solutions >= self._min_num_satisfying_solutions:
                    return best_satisfying_sol
            self._rng.shuffle(sampler_idxs)
            for sample_idx in sampler_idxs:
                sampler = samplers[sample_idx]
                partial_sol = sampler.sample(sol, self._rng)
                if partial_sol is not None:
                    break
            else:
                raise RuntimeError("All samplers produced None; solver stuck.")
            sol = sol.copy()
            sol.update(partial_sol)
        return best_satisfying_sol
