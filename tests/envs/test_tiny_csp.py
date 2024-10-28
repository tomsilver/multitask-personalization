"""Tests for tiny_csp.py."""

import numpy as np

from multitask_personalization.envs.tiny.tiny_csp import (
    TinyUserConstraint,
    create_tiny_csp,
)
from multitask_personalization.envs.tiny.tiny_env import (
    TinyEnv,
    TinyHiddenSpec,
    TinyState,
)
from multitask_personalization.utils import solve_csp


def test_pybullet_csp():
    """Tests for pybullet_skills.py."""
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
    human_position = obs.human

    # Create the CSP.
    csp, samplers, policy, initialization = create_tiny_csp(human_position, seed)
    # Use ground-truth parameters for constraint.
    assert len(csp.constraints) == 1
    constraint = csp.constraints[0]
    assert isinstance(constraint, TinyUserConstraint)
    constraint._desired_distance = desired_distance  # pylint: disable=protected-access
    constraint._distance_threshold = (  # pylint: disable=protected-access
        distance_threshold
    )

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
