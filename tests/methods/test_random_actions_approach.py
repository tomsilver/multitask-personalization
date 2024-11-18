"""Tests for random_actions_approach.py."""

from multitask_personalization.envs.tiny.tiny_env import (
    TinyEnv,
    TinyHiddenSpec,
    TinySceneSpec,
    TinyState,
)
from multitask_personalization.methods.random_actions_approach import (
    RandomActionsApproach,
)


def test_random_actions_approach():
    """Tests for random_actions_approach.py."""
    seed = 123
    scene_spec = TinySceneSpec()
    hidden_spec = TinyHiddenSpec(0.1, 0.01)
    env = TinyEnv(scene_spec, hidden_spec=hidden_spec, seed=seed)
    approach = RandomActionsApproach(scene_spec, env.action_space, seed=seed)
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
        assert not truncated

    env.close()
