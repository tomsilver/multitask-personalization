"""Tests for csp_approach.py."""

from multitask_personalization.envs.tiny.tiny_csp import TinyCSPGenerator
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

    # Run enough time steps to learn reasonable constraints.
    obs, info = env.reset()
    approach.reset(obs, info)
    for _ in range(10000):
        act = approach.step()
        obs, reward, terminated, truncated, info = env.step(act)
        approach.update(obs, reward, terminated, info)
        assert not truncated
        assert not terminated

    # pylint: disable=protected-access
    csp_generator = approach._csp_generator
    assert isinstance(csp_generator, TinyCSPGenerator)
    learned_dist = csp_generator._distance_constraint_generator._desired_distance
    assert learned_dist <= 1.5
    env.close()
