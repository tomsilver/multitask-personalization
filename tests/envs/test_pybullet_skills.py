"""Tests for pybullet_skills.py."""

import os
from pathlib import Path

import numpy as np
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import sample_collision_free_inverse_kinematics
from pybullet_helpers.link import get_link_pose
from tomsutils.llm import OpenAILLM

from multitask_personalization.envs.pybullet.pybullet_env import PyBulletEnv
from multitask_personalization.envs.pybullet.pybullet_scene_spec import (
    HiddenSceneSpec,
    PyBulletSceneSpec,
)
from multitask_personalization.envs.pybullet.pybullet_skills import (
    get_duster_head_frame_wiping_plan,
    get_plan_to_pick_object,
    get_plan_to_wipe_surface,
)
from multitask_personalization.envs.pybullet.pybullet_structs import (
    PyBulletAction,
    PyBulletState,
)
from multitask_personalization.envs.pybullet.pybullet_utils import PyBulletCannedLLM
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
    scene_spec = PyBulletSceneSpec(num_books=3)
    llm = OpenAILLM(
        model_name="gpt-4o-mini",
        cache_dir=Path(__file__).parents[1] / "unit_test_llm_cache",
        max_tokens=700,
        use_cache_only=True,
    )
    book_preferences = "I like pretty much anything!"
    rom_model = SphericalROMModel(scene_spec.human_spec)
    surfaces_robot_can_clean = [
        ("table", -1),
        ("shelf", 0),
        ("shelf", 1),
        ("shelf", 2),
    ]
    hidden_spec = HiddenSceneSpec(
        missions="all",
        book_preferences=book_preferences,
        rom_model=rom_model,
        surfaces_robot_can_clean=surfaces_robot_can_clean,
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
    _, book1 = obs.book_descriptions[:2]

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

    env.close()


def test_wiping_all_surfaces():
    """Tests for get_plan_to_wipe_surface() on all relevant surfaces."""
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "NOT A REAL KEY"  # will not be used

    seed = 123
    # NOTE: disable books.
    scene_spec = PyBulletSceneSpec(num_books=0)
    llm = PyBulletCannedLLM(
        cache_dir=Path(__file__).parents[1] / "unit_test_llm_cache",
    )
    book_preferences = "I like pretty much anything!"
    rom_model = SphericalROMModel(scene_spec.human_spec)
    surfaces_robot_can_clean = [
        ("table", -1),
        ("shelf", 0),
        ("shelf", 1),
        ("shelf", 2),
    ]
    hidden_spec = HiddenSceneSpec(
        missions="all",
        book_preferences=book_preferences,
        rom_model=rom_model,
        surfaces_robot_can_clean=surfaces_robot_can_clean,
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

    # Pick the duster.
    grasp_pose = scene_spec.duster_grasp
    pick_duster_plan = get_plan_to_pick_object(
        obs,
        "duster",
        grasp_pose,
        sim,
    )
    obs = _run_plan(pick_duster_plan, env)
    assert obs.held_object == "duster"

    # Wipe multiple surfaces.
    assert env.shelf_link_ids == {0, 1, 2}  # max is the "ceiling"
    targets = [
        ("shelf", 2, 0),
        ("shelf", 1, 0),
        ("shelf", 0, 0),
        ("table", -1, 1),
    ]
    rng = np.random.default_rng(123)
    sim.set_state(obs)
    world_to_duster_head = get_link_pose(
        sim.duster_id, sim.duster_head_link_id, sim.physics_client_id
    )
    world_to_ee = sim.robot.get_end_effector_pose()
    ee_to_duster_head = multiply_poses(world_to_ee.invert(), world_to_duster_head)
    collision_ids = sim.get_collision_ids() - {sim.current_held_object_id}
    for surface_name, link_id, num_rots in targets:
        # Sample base poses and starting joint states until one works.
        # Start by determining the initial end effector pose.
        duster_head_plan = get_duster_head_frame_wiping_plan(
            obs, "duster", surface_name, num_rots, sim, surface_link_id=link_id
        )
        ee_init_pose = multiply_poses(duster_head_plan[0], ee_to_duster_head.invert())
        robot_base_pose = None
        robot_joint_state = None
        for _ in range(1000):
            sim.set_state(obs)
            # Sample base pose.
            dx, dy = rng.uniform([-0.1, -0.1], [0.1, 0.1])
            position = (
                scene_spec.robot_base_pose.position[0] + dx,
                scene_spec.robot_base_pose.position[1] + dy,
                scene_spec.robot_base_pose.position[2],
            )
            orientation = scene_spec.robot_base_pose.orientation
            base_pose_candidate = Pose(position, orientation)
            # Sample joint state.
            sim.robot.set_base(base_pose_candidate)
            try:
                joint_state_candidate = next(
                    sample_collision_free_inverse_kinematics(
                        sim.robot,
                        ee_init_pose,
                        collision_ids,
                        rng,
                        held_object=sim.current_held_object_id,
                        base_link_to_held_obj=sim.current_grasp_transform,
                    )
                )
            except StopIteration:
                continue
            wipe_plan = get_plan_to_wipe_surface(
                obs,
                "duster",
                surface_name,
                base_pose_candidate,
                base_pose_candidate,
                joint_state_candidate,
                num_rots,
                sim,
                surface_link_id=link_id,
            )
            if wipe_plan is not None:
                robot_base_pose = base_pose_candidate
                robot_joint_state = joint_state_candidate
                break

        assert robot_base_pose is not None
        assert robot_joint_state is not None
        wipe_plan = get_plan_to_wipe_surface(
            obs,
            "duster",
            surface_name,
            robot_base_pose,
            robot_base_pose,
            robot_joint_state,
            num_rots,
            sim,
            surface_link_id=link_id,
        )
        obs = _run_plan(wipe_plan, env)
        sim.set_state(obs)

    env.close()
