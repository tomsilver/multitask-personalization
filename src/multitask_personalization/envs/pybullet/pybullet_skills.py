"""Python programs that implement various behaviors in PyBullet envs."""

from typing import Iterator

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, get_pose
from pybullet_helpers.inverse_kinematics import (
    check_body_collisions,
)
from pybullet_helpers.manipulation import (
    get_kinematic_plan_to_pick_object,
    get_kinematic_plan_to_place_object,
)
from pybullet_helpers.math_utils import get_poses_facing_line
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
from multitask_personalization.rom.models import (
    ROMModel,
)


def get_object_id_from_name(object_name: str, sim: PyBulletEnv) -> int:
    """Get the PyBullet ID in the given sim env from a name."""
    if object_name.startswith("book"):
        idx = int(object_name[len("book") :])
        return sim.book_ids[idx]
    return {
        "cup": sim.cup_id,
        "table": sim.table_id,
        "tray": sim.tray_id,
        "shelf": sim.shelf_id,
    }[object_name]


def get_surface_ids(sim: PyBulletEnv) -> set[int]:
    """Get all possible surfaces in the simulator."""
    surface_names = ["table", "tray", "shelf"]
    return {get_object_id_from_name(n, sim) for n in surface_names}


def get_surface_that_object_is_on(
    object_id: int, sim: PyBulletEnv, distance_threshold: float = 1e-3
) -> int:
    """Get the PyBullet ID of the surface that the object is on."""
    surfaces = get_surface_ids(sim)
    assert object_id not in surfaces
    object_pose = get_pose(object_id, sim.physics_client_id)
    for surface_id in surfaces:
        surface_pose = get_pose(surface_id, sim.physics_client_id)
        # Check if object pose is above surface pose.
        if object_pose.position[2] < surface_pose.position[2]:
            continue
        # Check for contact.
        if check_body_collisions(
            object_id,
            surface_id,
            sim.physics_client_id,
            distance_threshold=distance_threshold,
        ):
            return surface_id
    raise ValueError(f"Object {object_id} not on any surface.")


def get_collision_ids(sim: PyBulletEnv) -> set[int]:
    """Get all collision IDs for a sim env."""
    return set(sim.book_ids) | {
        sim.table_id,
        sim.human.body,
        sim.wheelchair.body,
        sim.shelf_id,
        sim.tray_id,
        sim.side_table_id,
    }


def generate_side_grasps(rng: np.random.Generator) -> Iterator[Pose]:
    """Generate side grasps."""
    while True:
        angle_offset = rng.uniform(-np.pi, np.pi)
        relative_pose = get_poses_facing_line(
            axis=(0.0, 0.0, 1.0),
            point_on_line=(0.0, 0.0, 0),
            radius=1e-3,
            num_points=1,
            angle_offset=angle_offset,
        )[0]
        yield relative_pose


def generate_surface_placements(
    surface_id: int, obj_id: int, sim: PyBulletEnv, rng: np.random.Generator
) -> Iterator[Pose]:
    """Sample placements uniformly on the top of the given surface."""
    surface_extents = get_aabb_dimensions(surface_id, sim)
    object_extents = get_aabb_dimensions(obj_id, sim)
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


def get_aabb_dimensions(object_id: int, sim: PyBulletEnv) -> tuple[float, float, float]:
    """Get the 3D bounding box dimensions of an object."""
    (min_x, min_y, min_z), (max_x, max_y, max_z) = p.getAABB(
        object_id, -1, sim.physics_client_id
    )
    return (max_x - min_x, max_y - min_y, max_z - min_z)


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
    obj_id = get_object_id_from_name(object_name, sim)
    surface_id = get_surface_that_object_is_on(obj_id, sim)
    collision_ids = get_collision_ids(sim) - {obj_id}
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
    object_id = get_object_id_from_name(object_name, sim)
    kinematic_state = get_kinematic_state_from_pybullet_state(state, sim)
    collision_ids = get_collision_ids(sim) - set(kinematic_state.attachments)
    surface_extents = get_aabb_dimensions(object_id, sim)

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
    object_id = get_object_id_from_name(object_name, sim)
    kinematic_state = get_kinematic_state_from_pybullet_state(state, sim)
    assert object_id in kinematic_state.attachments
    collision_ids = get_collision_ids(sim) - set(kinematic_state.attachments)

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


def sample_handover_pose(rom_model: ROMModel, rng: np.random.Generator) -> Pose:
    """Sample a candidate handover pose that is within the ROM."""
    position = tuple(rom_model.sample_reachable_position(rng))
    orientation = (
        0.8522037863731384,
        0.4745013415813446,
        -0.01094298530369997,
        0.22017613053321838,
    )
    pose = Pose(position, orientation)
    return pose


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
    object_id = get_object_id_from_name(object_name, sim)
    surface_id = get_object_id_from_name(surface_name, sim)
    collision_ids = get_collision_ids(sim) - {object_id}
    placement_generator = generate_surface_placements(surface_id, object_id, sim, rng)
    kinematic_state = get_kinematic_state_from_pybullet_state(state, sim)
    object_extents = get_aabb_dimensions(object_id, sim)
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
