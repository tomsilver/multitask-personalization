"""Tests for csp_solvers.py."""

import gymnasium as gym
import numpy as np
from tomsutils.spaces import EnumSpace

from multitask_personalization.csp_solvers import (
    ExhaustiveDiscreteCSPSolver,
    RandomWalkCSPSolver,
)
from multitask_personalization.structs import (
    CSP,
    CSPCost,
    CSPVariable,
    DiscreteCSP,
    FunctionalCSPConstraint,
    FunctionalCSPSampler,
    LogProbCSPConstraint,
)


def test_hybrid_csp_solvers():
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
    solver = RandomWalkCSPSolver(seed=123, show_progress_bar=False)
    sol = solver.solve(csp, initialization, samplers)
    assert sol is not None
    assert sol[x] < sol[y]
    assert sol[y] < sol[z] / 5


def test_discrete_csp_solvers():
    """Tests for discrete CSP solvers."""
    a = CSPVariable("a", EnumSpace(list(range(8))))
    b = CSPVariable("b", EnumSpace(list(range(8))))
    c = CSPVariable("c", EnumSpace(list(range(8))))

    # Optimal: a = 2, b = 3, c = 0.
    c1 = FunctionalCSPConstraint("c1", [a, b], lambda a, b: a + b <= 5)
    c2 = FunctionalCSPConstraint("c2", [a, b], lambda a, b: a < b)
    c3 = FunctionalCSPConstraint("c3", [c], lambda c: c <= 5)
    cost = CSPCost("cost", [a, b, c], lambda a, b, c: -(a * b * (10 - c)))

    csp = DiscreteCSP([a, b, c], [c1, c2, c3], cost)

    solver = ExhaustiveDiscreteCSPSolver(show_progress_bar=False)
    sol = solver.solve(csp)
    assert sol is not None
    assert sol[a] == 2
    assert sol[b] == 3
    assert sol[c] == 0
