"""Python programs that implement various behaviors in PyBullet envs."""

from typing import Iterator

import numpy as np
from pybullet_helpers.geometry import Pose, get_pose
from pybullet_helpers.manipulation import (
    get_kinematic_plan_to_pick_object,
    get_kinematic_plan_to_place_object,
)
from pybullet_helpers.motion_planning import (
    run_base_motion_planning,
    run_smooth_motion_planning_to_pose,
)
from pybullet_helpers.states import KinematicState

from multitask_personalization.envs.pybullet.pybullet_env import (
    PyBulletEnv,
)
from multitask_personalization.envs.pybullet.pybullet_structs import (
    GripperAction,
    PyBulletAction,
    PyBulletState,
)


def generate_surface_placements(
    surface_id: int, obj_id: int, sim: PyBulletEnv, rng: np.random.Generator
) -> Iterator[Pose]:
    """Sample placements uniformly on the top of the given surface."""
    # NOTE: this function currently assumes that the surface is a cube and that
    # the local frame is at the center of the cube.
    surface_extents = sim.get_aabb_dimensions(surface_id)
    object_extents = sim.get_aabb_dimensions(obj_id)
    placement_lb = (
        -surface_extents[0] / 2 + object_extents[0] / 2,
        -surface_extents[1] / 2 + object_extents[1] / 2,
        surface_extents[2] / 2 + object_extents[2] / 2,
    )
    placement_ub = (
        surface_extents[0] / 2 - object_extents[0] / 2,
        surface_extents[1] / 2 - object_extents[1] / 2,
        surface_extents[2] / 2 + object_extents[2] / 2,
    )

    while True:
        yield Pose(tuple(rng.uniform(placement_lb, placement_ub)))


def get_pybullet_action_plan_from_kinematic_plan(
    kinematic_plan: list[KinematicState],
) -> list[PyBulletAction]:
    """Convert a kinematic plan into a pybullet action plan."""
    action_plan: list[PyBulletAction] = []
    for s0, s1 in zip(kinematic_plan[:-1], kinematic_plan[1:], strict=True):
        actions = get_actions_from_kinematic_transition(s0, s1)
        action_plan.extend(actions)
    return action_plan


def get_kinematic_state_from_pybullet_state(
    pybullet_state: PyBulletState, sim: PyBulletEnv
) -> KinematicState:
    """Convert a PyBulletState into a KinematicState."""
    robot_joints = pybullet_state.robot_joints
    object_poses = {
        sim.cup_id: pybullet_state.cup_pose,
        sim.table_id: sim.task_spec.table_pose,
        sim.shelf_id: sim.task_spec.shelf_pose,
        sim.tray_id: sim.task_spec.tray_pose,
    }
    for book_id, book_pose in zip(sim.book_ids, pybullet_state.book_poses, strict=True):
        object_poses[book_id] = book_pose
    attachments: dict[int, Pose] = {}
    if pybullet_state.held_object == "cup":
        assert pybullet_state.grasp_transform is not None
        attachments[sim.cup_id] = pybullet_state.grasp_transform
    for book_idx, book_id in enumerate(sim.book_ids):
        if pybullet_state.held_object == f"book{book_idx}":
            assert pybullet_state.grasp_transform is not None
            attachments[book_id] = pybullet_state.grasp_transform
    return KinematicState(
        robot_joints, object_poses, attachments, pybullet_state.robot_base
    )


def get_actions_from_kinematic_transition(
    state: KinematicState, next_state: KinematicState
) -> list[PyBulletAction]:
    """Convert a single kinematic state transition into one or more actions."""
    assert state.robot_base_pose is not None
    assert next_state.robot_base_pose is not None
    base_delta = (
        next_state.robot_base_pose.position[0] - state.robot_base_pose.position[0],
        next_state.robot_base_pose.position[1] - state.robot_base_pose.position[1],
        next_state.robot_base_pose.rpy[2] - state.robot_base_pose.rpy[2],
    )
    joint_delta = np.subtract(next_state.robot_joints, state.robot_joints)
    delta = list(base_delta) + list(joint_delta[:7])
    actions: list[PyBulletAction] = [(0, delta)]
    if next_state.attachments and not state.attachments:
        actions.append((1, GripperAction.CLOSE))
    elif state.attachments and not next_state.attachments:
        actions.append((1, GripperAction.OPEN))
    return actions


def get_plan_to_pick_object(
    state: PyBulletState,
    object_name: str,
    grasp_pose: Pose,
    sim: PyBulletEnv,
    max_motion_planning_time: float = 1.0,
) -> list[PyBulletAction]:
    """Get a plan to pick up an object from some current state."""
    sim.set_state(state)
    obj_id = sim.get_object_id_from_name(object_name)
    surface_id = sim.get_surface_that_object_is_on(obj_id)
    collision_ids = sim.get_collision_ids() - {obj_id}
    grasp_generator = iter([grasp_pose])
    kinematic_state = get_kinematic_state_from_pybullet_state(state, sim)
    kinematic_plan = get_kinematic_plan_to_pick_object(
        kinematic_state,
        sim.robot,
        obj_id,
        surface_id,
        collision_ids,
        grasp_generator=grasp_generator,
        max_motion_planning_time=max_motion_planning_time,
    )
    assert kinematic_plan is not None
    return get_pybullet_action_plan_from_kinematic_plan(kinematic_plan)


def get_plan_to_move_next_to_object(
    state: PyBulletState,
    object_name: str,
    sim: PyBulletEnv,
    seed: int = 0,
) -> list[PyBulletAction]:
    """Get a plan to move next to a given object."""
    sim.set_state(state)
    object_id = sim.get_object_id_from_name(object_name)
    kinematic_state = get_kinematic_state_from_pybullet_state(state, sim)
    collision_ids = sim.get_collision_ids() - set(kinematic_state.attachments)
    surface_extents = sim.get_aabb_dimensions(object_id)

    current_base_pose = state.robot_base
    object_pose = get_pose(object_id, sim.physics_client_id)

    # Use pre-defined staging base poses for now. Generalize this later.
    if object_name == "tray":
        target_base_pose = Pose(
            (
                object_pose.position[0] - surface_extents[0],
                object_pose.position[1] - surface_extents[1],
                0.0,
            ),
            orientation=current_base_pose.orientation,
        )
    elif object_name == "shelf":
        target_base_pose = sim.task_spec.robot_base_pose  # initial base pose
    elif object_name == "table":
        target_base_pose = Pose(
            (
                sim.task_spec.robot_base_pose.position[0],
                sim.task_spec.robot_base_pose.position[1] - 0.1,
                sim.task_spec.robot_base_pose.position[2],
            ),
            sim.task_spec.robot_base_pose.orientation,
        )
    else:
        raise NotImplementedError

    if kinematic_state.attachments:
        assert len(kinematic_state.attachments) == 1
        held_obj_id, held_obj_tf = next(iter(kinematic_state.attachments.items()))
    else:
        held_obj_id, held_obj_tf = None, None

    base_motion_plan = run_base_motion_planning(
        sim.robot,
        current_base_pose,
        target_base_pose,
        position_lower_bounds=sim.task_spec.world_lower_bounds[:2],
        position_upper_bounds=sim.task_spec.world_upper_bounds[:2],
        collision_bodies=collision_ids,
        seed=seed,
        physics_client_id=sim.physics_client_id,
        platform=sim.robot_stand_id,
        held_object=held_obj_id,
        base_link_to_held_obj=held_obj_tf,
    )

    assert base_motion_plan is not None

    kinematic_plan: list[KinematicState] = []
    for base_pose in base_motion_plan:
        kinematic_plan.append(kinematic_state.copy_with(robot_base_pose=base_pose))

    return get_pybullet_action_plan_from_kinematic_plan(kinematic_plan)


def get_plan_to_handover_object(
    state: PyBulletState,
    object_name: str,
    handover_pose: Pose,
    sim: PyBulletEnv,
    seed: int = 0,
    max_motion_planning_time: float = 1.0,
) -> list[PyBulletAction]:
    """Get a plan to hand over a held object while next to a person."""
    sim.set_state(state)
    object_id = sim.get_object_id_from_name(object_name)
    kinematic_state = get_kinematic_state_from_pybullet_state(state, sim)
    assert object_id in kinematic_state.attachments
    collision_ids = sim.get_collision_ids() - set(kinematic_state.attachments)

    # Motion plan to hand over.
    kinematic_state.set_pybullet(sim.robot)
    robot_joint_plan = run_smooth_motion_planning_to_pose(
        handover_pose,
        sim.robot,
        collision_ids=collision_ids,
        end_effector_frame_to_plan_frame=Pose.identity(),
        seed=seed,
        max_time=max_motion_planning_time,
        held_object=object_id,
        base_link_to_held_obj=kinematic_state.attachments[object_id],
    )
    assert robot_joint_plan is not None
    kinematic_plan: list[KinematicState] = []
    for robot_joints in robot_joint_plan:
        kinematic_plan.append(kinematic_state.copy_with(robot_joints=robot_joints))

    return get_pybullet_action_plan_from_kinematic_plan(kinematic_plan)


def get_plan_to_place_object(
    state: PyBulletState,
    object_name: str,
    surface_name: str,
    sim: PyBulletEnv,
    rng: np.random.Generator,
    max_motion_planning_time: float = 1.0,
) -> list[PyBulletAction]:
    """Get a plan to place a held object on a given surface."""
    sim.set_state(state)
    object_id = sim.get_object_id_from_name(object_name)
    surface_id = sim.get_object_id_from_name(surface_name)
    collision_ids = sim.get_collision_ids() - {object_id}
    placement_generator = generate_surface_placements(surface_id, object_id, sim, rng)
    kinematic_state = get_kinematic_state_from_pybullet_state(state, sim)
    object_extents = sim.get_aabb_dimensions(object_id)
    kinematic_plan = get_kinematic_plan_to_place_object(
        kinematic_state,
        sim.robot,
        object_id,
        surface_id,
        collision_ids,
        placement_generator,
        preplace_translation_magnitude=object_extents[2],
        max_motion_planning_time=max_motion_planning_time,
    )
    assert kinematic_plan is not None
    return get_pybullet_action_plan_from_kinematic_plan(kinematic_plan)
