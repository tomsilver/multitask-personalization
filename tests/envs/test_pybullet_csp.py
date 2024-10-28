"""Tests for pybullet_csp.py."""

import numpy as np

from multitask_personalization.envs.pybullet.pybullet_csp import (
    create_book_handover_csp,
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
    preferred_books = ["book2"]
    rom_model = SphericalROMModel(task_spec.human_spec)
    hidden_spec = HiddenTaskSpec(book_preferences=preferred_books, rom_model=rom_model)

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
    csp, samplers, policy, initialization = create_book_handover_csp(
        sim,
        rom_model,
        preferred_books,
        seed,
        max_motion_planning_time=0.1,
    )

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
