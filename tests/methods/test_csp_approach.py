"""Tests for csp_approach.py."""

import pytest

from multitask_personalization.envs.tiny.tiny_csp import TinyCSPGenerator
from multitask_personalization.envs.tiny.tiny_env import (
    TinyEnv,
    TinyHiddenSpec,
)
from multitask_personalization.methods.csp_approach import (
    CSPApproach,
)


@pytest.mark.parametrize("explore_method", ["nothing-personal", "ensemble"])
def test_csp_approach(explore_method):
    """Tests for csp_approach.py."""
    seed = 123

    hidden_spec = TinyHiddenSpec(1.0, 0.5)
    env = TinyEnv(hidden_spec=hidden_spec, seed=seed)
    approach = CSPApproach(env.action_space, seed=seed, explore_method=explore_method)
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

    csp_generator = approach._csp_generator  # pylint: disable=protected-access
    assert isinstance(csp_generator, TinyCSPGenerator)
    learned_dist = (
        csp_generator._distance_constraint_generator._desired_distance  # pylint: disable=protected-access
    )
    assert learned_dist <= 1.1

    env.close()
