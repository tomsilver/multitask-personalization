"""Tests for pybullet_skills.py."""

import numpy as np

from multitask_personalization.envs.pybullet.pybullet_env import PyBulletEnv
from multitask_personalization.envs.pybullet.pybullet_skills import (
    get_plan_to_move_next_to_object,
    get_plan_to_pick_object,
    get_plan_to_place_object,
)
from multitask_personalization.envs.pybullet.pybullet_structs import (
    PyBulletAction,
    PyBulletState,
)
from multitask_personalization.envs.pybullet.pybullet_task_spec import PyBulletTaskSpec


def _run_plan(plan: list[PyBulletAction], env: PyBulletEnv) -> PyBulletState:
    for act in plan:
        obs, reward, terminated, truncated, _ = env.step(act)
        assert isinstance(obs, PyBulletState)
        assert reward >= 0
        assert not terminated
        assert not truncated
    return obs


def test_pybullet_skills():
    """Tests for pybullet_skills.py."""
    seed = 123
    rng = np.random.default_rng(seed)
    task_spec = PyBulletTaskSpec()

    # Create a real environment.
    env = PyBulletEnv(task_spec, use_gui=False, seed=seed)

    # Uncomment to create video.
    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/test-pybullet-skills")

    env.action_space.seed(seed)
    obs, _ = env.reset()
    assert isinstance(obs, PyBulletState)

    # Create a simulator.
    sim = PyBulletEnv(task_spec, use_gui=False, seed=seed)

    # Test pick book.
    pick_book_plan = get_plan_to_pick_object(obs, "book1", sim, rng)
    obs = _run_plan(pick_book_plan, env)
    assert obs.held_object == "book1"

    # Test move to tray.
    move_to_tray_plan = get_plan_to_move_next_to_object(obs, "tray", sim, seed=seed)
    obs = _run_plan(move_to_tray_plan, env)

    # Test place book on tray.
    place_book_on_tray_plan = get_plan_to_place_object(obs, "book1", "tray", sim, rng)
    obs = _run_plan(place_book_on_tray_plan, env)
    assert obs.held_object is None

    # Test move to shelf.
    move_to_shelf_plan = get_plan_to_move_next_to_object(obs, "shelf", sim, seed=seed)
    obs = _run_plan(move_to_shelf_plan, env)

    # Test pick another book.
    pick_book_plan = get_plan_to_pick_object(obs, "book0", sim, rng)
    obs = _run_plan(pick_book_plan, env)
    assert obs.held_object == "book0"

    # Test move to tray.
    move_to_tray_plan = get_plan_to_move_next_to_object(obs, "tray", sim, seed=seed)
    obs = _run_plan(move_to_tray_plan, env)

    # Test place book on tray.
    place_book_on_tray_plan = get_plan_to_place_object(obs, "book0", "tray", sim, rng)
    obs = _run_plan(place_book_on_tray_plan, env)
    assert obs.held_object is None

    env.close()
