"""Tests for csp_solvers.py."""

import gymnasium as gym
import numpy as np

from multitask_personalization.csp_solvers import (
    LifelongCSPSolverWrapper,
    RandomWalkCSPSolver,
)
from multitask_personalization.structs import (
    CSP,
    CSPVariable,
    FunctionalCSPConstraint,
    FunctionalCSPSampler,
    LogProbCSPConstraint,
)


def _create_test_csp():
    x = CSPVariable("x", gym.spaces.Box(0, 1, dtype=np.float_))
    y = CSPVariable("y", gym.spaces.Box(0, 1, dtype=np.float_))
    z = CSPVariable("z", gym.spaces.Discrete(5))

    c1 = FunctionalCSPConstraint("c1", [x, y], lambda x, y: x < y)
    c2 = LogProbCSPConstraint("c2", [y, z], lambda y, z: np.log(y < z / 5))

    csp = CSP([x, y, z], [c1, c2])

    sample_xy = lambda _, rng: {
        x: rng.uniform(0, 1, size=(1,)),
        y: rng.uniform(0, 1, size=(1,)),
    }
    sample_z = lambda _, rng: {z: rng.integers(5)}

    sampler_xy = FunctionalCSPSampler(sample_xy, csp, {x, y})
    sampler_z = FunctionalCSPSampler(sample_z, csp, {z})
    samplers = [sampler_xy, sampler_z]
    initialization = {x: 0.0, y: 0.0, z: 0}

    return csp, initialization, samplers


def test_solve_csp():
    """Tests for csp_solvers.py."""

    # Test RandomWalkCSPSolver().
    csp, initialization, samplers = _create_test_csp()
    solver = RandomWalkCSPSolver(seed=123, show_progress_bar=False)
    sol = solver.solve(csp, initialization, samplers)
    assert sol is not None

    # Test LifelongCSPSolverWrapper(RandomWalkCSPSolver()).
    # The lifelong solver should still work after deleting the samplers because
    # it should use its own memory-based samplers.
    lifelong_solver = LifelongCSPSolverWrapper(solver, seed=123)
    sol = lifelong_solver.solve(csp, initialization, samplers)
    assert sol is not None
    # Regenerate the CSP to make sure that equality checking is based on names.
    csp, initialization, samplers = _create_test_csp()
    sol = lifelong_solver.solve(csp, initialization, [])
    assert sol is not None
