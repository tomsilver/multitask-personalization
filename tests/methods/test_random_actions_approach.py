"""Tests for random_actions_approach.py."""

from multitask_personalization.envs.pybullet.pybullet_env import PyBulletEnv
from multitask_personalization.envs.pybullet.pybullet_structs import PyBulletState
from multitask_personalization.envs.pybullet.pybullet_task_spec import (
    HiddenTaskSpec,
    PyBulletTaskSpec,
)
from multitask_personalization.methods.random_actions_approach import (
    RandomActionsApproach,
)
from multitask_personalization.rom.models import SphericalROMModel


def test_random_actions_approach():
    """Tests for random_actions_approach.py."""
    seed = 123

    task_spec = PyBulletTaskSpec()
    preferred_books = ["book2"]
    rom_model = SphericalROMModel(task_spec.human_spec)
    hidden_spec = HiddenTaskSpec(book_preferences=preferred_books, rom_model=rom_model)
    env = PyBulletEnv(task_spec, hidden_spec=hidden_spec, use_gui=False, seed=seed)

    # Uncomment to make videos.
    # from gym.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/pybullet-random-actions-test")

    approach = RandomActionsApproach(env.action_space, seed=seed)
    approach.eval()
    env.action_space.seed(seed)
    obs, _ = env.reset()
    approach.reset(obs)
    assert isinstance(obs, PyBulletState)

    for _ in range(10):
        act = approach.step()
        obs, reward, terminated, truncated, _ = env.step(act)
        approach.update(obs, reward, terminated)
        assert isinstance(obs, PyBulletState)
        assert reward >= 0
        assert not terminated
        assert not truncated

    env.close()
