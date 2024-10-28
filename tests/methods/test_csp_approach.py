"""Tests for csp_approach.py."""

from multitask_personalization.envs.tiny.tiny_env import (
    TinyEnv,
    TinyHiddenSpec,
    TinyState,
)
from multitask_personalization.methods.csp_approach import (
    CSPApproach,
)


def test_csp_approach():
    """Tests for csp_approach.py."""
    seed = 123

    hidden_spec = TinyHiddenSpec(0.1, 0.01)
    env = TinyEnv(hidden_spec=hidden_spec, seed=seed)
    approach = CSPApproach(env.action_space, seed=seed)
    approach.eval()
    env.action_space.seed(seed)
    obs, info = env.reset()
    approach.reset(obs, info)
    assert isinstance(obs, TinyState)

    for _ in range(10):
        act = approach.step()
        obs, reward, terminated, truncated, info = env.step(act)
        approach.update(obs, reward, terminated, info)
        assert isinstance(obs, TinyState)
        assert not terminated
        assert not truncated

    env.close()
