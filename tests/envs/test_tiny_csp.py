"""Tests for tiny_csp.py."""

import numpy as np

from multitask_personalization.csp_solvers import RandomWalkCSPSolver
from multitask_personalization.envs.tiny.tiny_csp import (
    TinyCSPGenerator,
)
from multitask_personalization.envs.tiny.tiny_env import (
    TinyEnv,
    TinyHiddenSpec,
    TinySceneSpec,
    TinyState,
)


def test_tiny_csp():
    """Tests for tiny_csp.py."""
    seed = 123
    desired_distance = 0.1
    distance_threshold = 0.01
    scene_spec = TinySceneSpec()
    hidden_spec = TinyHiddenSpec(
        desired_distance=desired_distance, distance_threshold=distance_threshold
    )
    env = TinyEnv(scene_spec, hidden_spec=hidden_spec, seed=seed)
    obs, _ = env.reset()
    assert isinstance(obs, TinyState)

    # Create the CSP.
    csp_generator = TinyCSPGenerator(seed=seed)
    csp, samplers, policy, initialization = csp_generator.generate(obs)

    # Solve the CSP.
    solver = RandomWalkCSPSolver(
        seed, min_num_satisfying_solutions=5, show_progress_bar=False
    )
    sol = solver.solve(
        csp,
        initialization,
        samplers,
    )
    assert sol is not None
    policy.reset(sol)

    # Run the policy.
    for _ in range(100):
        act = policy.step(obs)
        obs, reward, env_terminated, truncated, _ = env.step(act)
        assert isinstance(obs, TinyState)
        assert np.isclose(reward, 0.0)
        assert not env_terminated
        assert not truncated

    env.close()
