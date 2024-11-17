"""Tests for csp_solvers.py."""

import gymnasium as gym
import numpy as np
import pytest
from tomsutils.spaces import EnumSpace

from multitask_personalization.csp_solvers import (
    BruteForceDiscreteCSPSolver,
    RandomWalkCSPSolver,
    TreeSearchIncrementalCSPSolver,
)
from multitask_personalization.structs import (
    CSP,
    CSPVariable,
    DiscreteCSP,
    FunctionalCSPConstraint,
    FunctionalCSPSampler,
    LogProbCSPConstraint,
)


@pytest.mark.parametrize(
    "solver",
    [
        RandomWalkCSPSolver(seed=123, show_progress_bar=False),
        TreeSearchIncrementalCSPSolver(
            seed=123,
            discrete_csp_solver=BruteForceDiscreteCSPSolver(
                seed=123, show_progress_bar=False
            ),
            max_depth=2,
            base_branching_factor=3,
            progressive_widening_scale=1.1,
        ),
    ],
)
def test_hybrid_csp_solvers(solver):
    """Tests for discrete-continuous CSP solvers."""
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
    sol = solver.solve(csp, initialization, samplers)
    assert sol is not None
    assert sol[x] < sol[y]
    assert sol[y] < sol[z] / 5


@pytest.mark.parametrize(
    "solver", [BruteForceDiscreteCSPSolver(seed=123, show_progress_bar=False)]
)
def test_discrete_csp_solvers(solver):
    """Tests for discrete CSP solvers."""
    a = CSPVariable("a", EnumSpace(list(range(8))))
    b = CSPVariable("b", EnumSpace(list(range(8))))

    # Optimal: a = 2, b = 3.
    c1 = FunctionalCSPConstraint("c1", [a, b], lambda a, b: a + b == 5)
    c2 = FunctionalCSPConstraint("c2", [a, b], lambda a, b: a < b)
    c3 = FunctionalCSPConstraint("c3", [a, b], lambda a, b: b - a == 1)

    csp = DiscreteCSP([a, b], [c1, c2, c3])

    sol = solver.solve(csp)
    assert sol is not None
    assert sol[a] == 2
    assert sol[b] == 3
