"""Tests for pybullet_csp.py."""

import numpy as np

from multitask_personalization.envs.pybullet.pybullet_csp import (
    create_book_handover_csp,
)
from multitask_personalization.envs.pybullet.pybullet_env import PyBulletEnv
from multitask_personalization.envs.pybullet.pybullet_structs import (
    PyBulletState,
)
from multitask_personalization.envs.pybullet.pybullet_task_spec import PyBulletTaskSpec
from multitask_personalization.rom.models import LearnedROMModel
from multitask_personalization.utils import solve_csp


def test_pybullet_csp():
    """Tests for pybullet_skills.py."""
    seed = 123
    rng = np.random.default_rng(seed)
    task_spec = PyBulletTaskSpec()

    # Create a real environment.
    env = PyBulletEnv(task_spec, use_gui=False, seed=seed)

    # Uncomment to create video.
    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/test-pybullet-csp")

    env.action_space.seed(seed)
    obs, _ = env.reset()
    assert isinstance(obs, PyBulletState)

    # Create a simulator.
    sim = PyBulletEnv(task_spec, use_gui=False, seed=seed)

    # Create the learned ROM model.
    rom_model = LearnedROMModel(0.1)
    rom_model.set_reachable_points(
        sim.create_reachable_position_cloud(rom_model.get_reachable_joints())
    )

    # Create book preferences.
    preferred_books = ["book2"]

    # Create the CSP.
    csp, samplers, policy, initialization = create_book_handover_csp(
        sim, rom_model, preferred_books, seed
    )

    # Solve the CSP.
    sol = solve_csp(csp, initialization, samplers, rng)
    policy.reset(sol)

    # Run the policy.
    # For now, just inspecting this visually; assertions coming soon.
    for _ in range(200):
        act = policy.step(obs)
        obs, reward, terminated, truncated, _ = env.step(act)
        assert isinstance(obs, PyBulletState)
        assert reward >= 0
        assert not terminated
        assert not truncated

    env.close()
