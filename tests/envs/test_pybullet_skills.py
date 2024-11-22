"""Tests for pybullet_skills.py."""

import os
from pathlib import Path

import numpy as np
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses
from tomsutils.llm import OpenAILLM

from multitask_personalization.envs.pybullet.pybullet_env import PyBulletEnv
from multitask_personalization.envs.pybullet.pybullet_scene_spec import (
    HiddenSceneSpec,
    PyBulletSceneSpec,
)
from multitask_personalization.envs.pybullet.pybullet_skills import (
    get_plan_to_move_next_to_object,
    get_plan_to_pick_object,
    get_plan_to_place_object,
    get_plan_to_wipe_surface,
)
from multitask_personalization.envs.pybullet.pybullet_structs import (
    PyBulletAction,
    PyBulletState,
)
from multitask_personalization.rom.models import SphericalROMModel


def _run_plan(plan: list[PyBulletAction], env: PyBulletEnv) -> PyBulletState:
    for act in plan:
        obs, reward, terminated, truncated, _ = env.step(act)
        assert isinstance(obs, PyBulletState)
        assert np.isclose(reward, 0.0)
        assert not terminated
        assert not truncated
    return obs


def test_pybullet_skills():
    """Tests for pybullet_skills.py."""
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "NOT A REAL KEY"  # will not be used

    seed = 123
    default_scene_spec = PyBulletSceneSpec()
    scene_spec = PyBulletSceneSpec(
        side_table_pose=Pose(position=(1.45, 0.0, -0.1)),
        book_half_extents=default_scene_spec.book_half_extents[:3],
        book_poses=default_scene_spec.book_poses[:3],
        book_rgbas=default_scene_spec.book_rgbas[:3],
    )
    llm = OpenAILLM(
        model_name="gpt-4o-mini",
        cache_dir=Path(__file__).parents[1] / "unit_test_llm_cache",
        max_tokens=700,
        use_cache_only=True,
    )
    book_preferences = "I like pretty much anything!"
    rom_model = SphericalROMModel(scene_spec.human_spec)
    hidden_spec = HiddenSceneSpec(
        book_preferences=book_preferences, rom_model=rom_model
    )

    # Create a real environment.
    env = PyBulletEnv(
        scene_spec,
        llm,
        hidden_spec=hidden_spec,
        use_gui=False,
        seed=seed,
    )

    # Uncomment to create video.
    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/test-pybullet-skills")

    env.action_space.seed(seed)
    obs, _ = env.reset()
    assert isinstance(obs, PyBulletState)

    # Create a simulator.
    sim = PyBulletEnv(scene_spec, llm, use_gui=False, seed=seed)
    book0, book1 = obs.book_descriptions[:2]

    # Extract the relative pose of the duster so we can later place it back.
    world_to_table = get_pose(sim.table_id, sim.physics_client_id)
    world_to_duster = get_pose(sim.duster_id, sim.physics_client_id)
    duster_placement_pose = multiply_poses(world_to_table.invert(), world_to_duster)

    # Test pick duster.
    grasp_pose = Pose.from_rpy(
        (
            scene_spec.duster_pole_offset[0] + 2 * scene_spec.duster_pole_radius,
            0,
            scene_spec.duster_head_half_extents[2] + scene_spec.duster_pole_height / 2,
        ),
        (np.pi / 2, np.pi, -np.pi / 2),
    )
    pick_duster_plan = get_plan_to_pick_object(
        obs,
        "duster",
        grasp_pose,
        sim,
    )
    obs = _run_plan(pick_duster_plan, env)
    assert obs.held_object == "duster"

    # Test wipe table with duster.
    wipe_direction_num_rotations = 1
    wipe_plan = get_plan_to_wipe_surface(
        obs,
        "duster",
        "table",
        wipe_direction_num_rotations,
        sim,
    )
    obs = _run_plan(wipe_plan, env)

    # Test place duster.
    place_duster_on_table_plan = get_plan_to_place_object(
        obs,
        "duster",
        "table",
        duster_placement_pose,
        sim,
    )
    obs = _run_plan(place_duster_on_table_plan, env)
    assert obs.held_object is None

    # Test pick book.
    grasp_pose = Pose((0, 0, 0), (-np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2))
    pick_book_plan = get_plan_to_pick_object(
        obs,
        book1,
        grasp_pose,
        sim,
    )
    obs = _run_plan(pick_book_plan, env)
    assert obs.held_object == book1

    # Test move to tray.
    move_to_tray_plan = get_plan_to_move_next_to_object(obs, "tray", sim, seed=seed)
    obs = _run_plan(move_to_tray_plan, env)

    # Test place book on tray.
    surface_extents = sim.get_aabb_dimensions(sim.tray_id)
    object_extents = sim.get_aabb_dimensions(sim.book_ids[1])
    placement_pose = Pose(
        (
            -surface_extents[0] / 2 + object_extents[0] / 2,
            0,
            surface_extents[2] / 2 + object_extents[2] / 2,
        )
    )
    place_book_on_tray_plan = get_plan_to_place_object(
        obs,
        book1,
        "tray",
        placement_pose,
        sim,
    )
    obs = _run_plan(place_book_on_tray_plan, env)
    assert obs.held_object is None

    # Test move to shelf.
    move_to_shelf_plan = get_plan_to_move_next_to_object(obs, "shelf", sim, seed=seed)
    obs = _run_plan(move_to_shelf_plan, env)

    # Test pick another book.
    grasp_pose = Pose((0, 0, 0), (-np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2))
    pick_book_plan = get_plan_to_pick_object(
        obs,
        book0,
        grasp_pose,
        sim,
    )
    obs = _run_plan(pick_book_plan, env)
    assert obs.held_object == book0

    env.close()
