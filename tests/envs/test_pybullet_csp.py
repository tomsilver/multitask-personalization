"""Tests for pybullet_csp.py."""

import numpy as np

from multitask_personalization.envs.pybullet.pybullet_csp import (
    PyBulletCSPGenerator,
)
from multitask_personalization.envs.pybullet.pybullet_env import PyBulletEnv
from multitask_personalization.envs.pybullet.pybullet_structs import (
    PyBulletState,
)
from multitask_personalization.envs.pybullet.pybullet_task_spec import (
    HiddenTaskSpec,
    PyBulletTaskSpec,
)
from multitask_personalization.rom.models import SphericalROMModel
from multitask_personalization.utils import solve_csp


def test_pybullet_csp():
    """Tests for pybullet_csp.py."""
    seed = 123
    rng = np.random.default_rng(seed)
    task_spec = PyBulletTaskSpec()
    book_preferences = "I like pretty much anything"
    rom_model = SphericalROMModel(task_spec.human_spec)
    hidden_spec = HiddenTaskSpec(book_preferences=book_preferences, rom_model=rom_model)

    # Create a real environment.
    env = PyBulletEnv(task_spec, hidden_spec=hidden_spec, use_gui=False, seed=seed)

    # Uncomment to create video.
    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/test-pybullet-csp")

    env.action_space.seed(seed)
    obs, _ = env.reset()
    assert isinstance(obs, PyBulletState)

    # Create a simulator.
    sim = PyBulletEnv(task_spec, use_gui=False, seed=seed)

    # Create the CSP.
    csp_generator = PyBulletCSPGenerator(
        sim,
        rom_model,
        seed,
    )
    csp, samplers, policy, initialization = csp_generator.generate(obs)

    # Solve the CSP.
    sol = solve_csp(csp, initialization, samplers, rng)
    policy.reset(sol)

    # Run the policy.
    # For now, just inspecting this visually; assertions coming soon.
    for _ in range(1000):
        act = policy.step(obs)
        obs, reward, terminated, truncated, _ = env.step(act)
        assert isinstance(obs, PyBulletState)
        if reward > 0:
            break
        assert not terminated
        assert not truncated
    else:
        assert False, "Policy did not terminate."

    env.close()
