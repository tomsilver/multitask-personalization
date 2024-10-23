"""Tests for pybullet_skills.py."""

import numpy as np
from pybullet_helpers.geometry import Pose

from multitask_personalization.envs.pybullet.pybullet_env import PyBulletEnv
from multitask_personalization.envs.pybullet.pybullet_skills import (
    get_plan_to_handover_object,
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
    grasp_pose = Pose((0, 0, 0), (-np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2))
    pick_book_plan = get_plan_to_pick_object(obs, "book1", grasp_pose, sim)
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
    grasp_pose = Pose((0, 0, 0), (-np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2))
    pick_book_plan = get_plan_to_pick_object(obs, "book0", grasp_pose, sim)
    obs = _run_plan(pick_book_plan, env)
    assert obs.held_object == "book0"

    # Test move to tray.
    move_to_tray_plan = get_plan_to_move_next_to_object(obs, "tray", sim, seed=seed)
    obs = _run_plan(move_to_tray_plan, env)

    # Test hand over book.
    handover_pose = Pose(
        (0.6096954345703125, 0.029336635023355484, 0.4117525517940521),
        (
            0.8522037863731384,
            0.4745013415813446,
            -0.01094298530369997,
            0.22017613053321838,
        ),
    )
    place_book_on_tray_plan = get_plan_to_handover_object(
        obs, "book0", handover_pose, sim, seed
    )
    obs = _run_plan(place_book_on_tray_plan, env)

    env.close()
