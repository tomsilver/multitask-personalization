"""Tests for csp_approach.py."""

import numpy as np
import pytest

from multitask_personalization.envs.tiny.tiny_csp import TinyCSPGenerator
from multitask_personalization.envs.tiny.tiny_env import (
    TinyEnv,
    TinyHiddenSpec,
)
from multitask_personalization.methods.csp_approach import (
    CSPApproach,
)


@pytest.mark.parametrize("explore_method", ["max-entropy", "nothing-personal"])
def test_csp_approach(explore_method):
    """Tests for csp_approach.py."""
    seed = 123

    hidden_spec = TinyHiddenSpec(1.0, 0.5)
    env = TinyEnv(hidden_spec=hidden_spec, seed=seed)
    csp_generator = TinyCSPGenerator(seed=seed, explore_method=explore_method)
    approach = CSPApproach(
        env.action_space,
        seed=seed,
        csp_generator=csp_generator,
        show_csp_progress_bar=False,
    )
    approach.train()
    env.action_space.seed(seed)

    for _ in range(10):
        obs, info = env.reset()
        approach.reset(obs, info)
        for _ in range(100):
            act = approach.step()
            obs, reward, terminated, truncated, info = env.step(act)
            assert np.isclose(reward, 0.0)
            approach.update(obs, reward, terminated, info)
            assert not truncated
            if terminated:
                break

    # pylint: disable=protected-access
    csp_generator = approach._csp_generator
    assert isinstance(csp_generator, TinyCSPGenerator)
    learned_dist = csp_generator._distance_constraint_generator._desired_distance
    assert learned_dist <= 1.5
    env.close()
