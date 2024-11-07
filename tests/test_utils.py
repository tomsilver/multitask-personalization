"""Tests for utils.py."""

import gymnasium as gym
import numpy as np

from multitask_personalization.structs import (
    CSP,
    CSPVariable,
    FunctionalCSPConstraint,
    FunctionalCSPSampler,
    LogProbCSPConstraint,
)
from multitask_personalization.utils import solve_csp


def test_solve_csp():
    """Tests for solve_csp()."""
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
    rng = np.random.default_rng(123)
    sol = solve_csp(csp, initialization, samplers, rng)
    assert sol is not None
    assert sol[x] < sol[y]
    assert sol[y] < sol[z] / 5
