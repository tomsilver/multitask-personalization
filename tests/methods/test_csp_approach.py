"""Tests for csp_approach.py."""

import numpy as np

from multitask_personalization.envs.tiny.tiny_csp import TinyUserConstraint
from multitask_personalization.envs.tiny.tiny_env import (
    TinyEnv,
    TinyHiddenSpec,
)
from multitask_personalization.methods.csp_approach import (
    CSPApproach,
)


def test_csp_approach():
    """Tests for csp_approach.py."""
    seed = 123

    hidden_spec = TinyHiddenSpec(1.0, 0.5)
    env = TinyEnv(hidden_spec=hidden_spec, seed=seed)
    approach = CSPApproach(env.action_space, seed=seed)
    approach.train()
    env.action_space.seed(seed)

    # Run enough episodes to learn reasonable constraints.
    for _ in range(500):
        obs, info = env.reset()
        approach.reset(obs, info)
        for _ in range(100):
            act = approach.step()
            obs, reward, terminated, truncated, info = env.step(act)
            approach.update(obs, reward, terminated, info)
            assert not truncated
            if terminated:
                break

    csp = approach._current_csp  # pylint: disable=protected-access
    constraint = csp.constraints[0]
    assert isinstance(constraint, TinyUserConstraint)
    learned_dist = constraint._desired_distance  # pylint: disable=protected-access
    assert np.isclose(learned_dist, 1.0, atol=1e-1)

    env.close()
