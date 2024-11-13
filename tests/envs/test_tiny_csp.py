"""Tests for tiny_csp.py."""

import numpy as np

from multitask_personalization.envs.tiny.tiny_csp import (
    TinyCSPGenerator,
)
from multitask_personalization.envs.tiny.tiny_env import (
    TinyEnv,
    TinyHiddenSpec,
    TinySceneSpec,
    TinyState,
)
from multitask_personalization.utils import solve_csp


def test_tiny_csp():
    """Tests for tiny_csp.py."""
    seed = 123
    rng = np.random.default_rng(seed)
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
    csp_generator = TinyCSPGenerator(
        seed=seed,
        distance_threshold=distance_threshold,
        init_desired_distance=desired_distance,
    )
    csp, samplers, policy, initialization = csp_generator.generate(obs)

    # Solve the CSP.
    sol = solve_csp(
        csp,
        initialization,
        samplers,
        rng,
        min_num_satisfying_solutions=5,
        show_progress_bar=False,
    )
    assert sol is not None
    policy.reset(sol)

    # Run the policy.
    for _ in range(100):
        act = policy.step(obs)
        obs, reward, env_terminated, truncated, _ = env.step(act)
        assert isinstance(obs, TinyState)
        if policy.check_termination(obs):
            assert np.isclose(reward, 0.0)
            break
        assert not env_terminated
        assert not truncated
    else:
        assert False, "Policy did not terminate."

    env.close()
