"""Tests for tiny_csp.py."""

import numpy as np

from multitask_personalization.envs.tiny.tiny_csp import (
    TinyCSPGenerator,
)
from multitask_personalization.envs.tiny.tiny_env import (
    TinyEnv,
    TinyHiddenSpec,
    TinyState,
)
from multitask_personalization.utils import solve_csp


def test_tiny_csp():
    """Tests for tiny_csp.py."""
    seed = 123
    rng = np.random.default_rng(seed)
    desired_distance = 0.1
    distance_threshold = 0.01
    hidden_spec = TinyHiddenSpec(
        desired_distance=desired_distance, distance_threshold=distance_threshold
    )
    env = TinyEnv(hidden_spec=hidden_spec, seed=seed)
    obs, _ = env.reset()
    assert isinstance(obs, TinyState)

    # Create the CSP.
    csp_generator = TinyCSPGenerator(
        seed,
        distance_threshold=distance_threshold,
        init_desired_distance=desired_distance,
    )
    csp, samplers, policy, initialization = csp_generator.generate(obs)

    # Solve the CSP.
    sol = solve_csp(csp, initialization, samplers, rng)
    policy.reset(sol)

    # Run the policy.
    for _ in range(100):
        act = policy.step(obs)
        obs, reward, terminated, truncated, _ = env.step(act)
        assert isinstance(obs, TinyState)
        if reward > 0:
            break
        assert not terminated
        assert not truncated
    else:
        assert False, "Policy did not terminate."

    env.close()
