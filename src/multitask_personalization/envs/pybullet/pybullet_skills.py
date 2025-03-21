"""Python programs that implement various behaviors in PyBullet envs."""

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, iter_between_poses, multiply_poses
from pybullet_helpers.inverse_kinematics import (
    check_body_collisions,
    check_collisions_with_held_object,
)
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.link import get_link_pose
from pybullet_helpers.manipulation import (
    get_kinematic_plan_to_pick_object,
    get_kinematic_plan_to_place_object,
    get_kinematic_plan_to_retract,
)
from pybullet_helpers.motion_planning import (
    MotionPlanningHyperparameters,
    run_base_motion_planning,
    run_motion_planning,
    run_smooth_motion_planning_to_pose,
)
from pybullet_helpers.states import KinematicState

from multitask_personalization.envs.pybullet.pybullet_env import PyBulletEnv
from multitask_personalization.envs.pybullet.pybullet_structs import (
    GripperAction,
    PyBulletAction,
    PyBulletState,
)


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
        sim.duster_id: pybullet_state.duster_pose,
        sim.table_id: sim.scene_spec.table_pose,
        sim.shelf_id: sim.scene_spec.shelf_pose,
    }
    for side_table_id, side_table_pose in zip(
        sim.side_table_ids, sim.scene_spec.side_table_poses, strict=True
    ):
        object_poses[side_table_id] = side_table_pose
    for book_id, book_pose in zip(sim.book_ids, pybullet_state.book_poses, strict=True):
        object_poses[book_id] = book_pose
    attachments: dict[int, Pose] = {}
    if pybullet_state.held_object == "cup":
        assert pybullet_state.grasp_transform is not None
        attachments[sim.cup_id] = pybullet_state.grasp_transform
    if pybullet_state.held_object == "duster":
        assert pybullet_state.grasp_transform is not None
        attachments[sim.duster_id] = pybullet_state.grasp_transform
    for book_id, book_description in zip(
        sim.book_ids, pybullet_state.book_descriptions, strict=True
    ):
        if pybullet_state.held_object == book_description:
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


def get_plan_to_retract(
    state: PyBulletState,
    sim: PyBulletEnv,
    collision_ids: set[int],
    translation_magnitude: float = 0.125,
    max_motion_planning_time: float = np.inf,
) -> list[PyBulletAction]:
    """Get a plan to retract in the opposite direction of the robot fingers."""
    sim.set_state(state)
    kinematic_state = get_kinematic_state_from_pybullet_state(state, sim)
    kinematic_plan = get_kinematic_plan_to_retract(
        kinematic_state,
        sim.robot,
        collision_ids,
        translation_magnitude=translation_magnitude,
        max_motion_planning_time=max_motion_planning_time,
        max_smoothing_iters_per_step=1,
    )
    assert kinematic_plan is not None
    return get_pybullet_action_plan_from_kinematic_plan(kinematic_plan)


def get_plan_to_pick_object(
    state: PyBulletState,
    object_name: str,
    grasp_pose: Pose,
    sim: PyBulletEnv,
    max_motion_planning_candidates: int = 1,
    max_motion_planning_time: float = np.inf,
) -> list[PyBulletAction] | None:
    """Get a plan to pick up an object from some current state."""
    sim.set_state(state)
    obj_id = sim.get_object_id_from_name(object_name)
    surface_id = sim.get_surface_that_object_is_on(obj_id)
    collision_ids = sim.get_collision_ids() - {obj_id}
    grasp_generator = iter([grasp_pose])
    kinematic_state = get_kinematic_state_from_pybullet_state(state, sim)
    kinematic_plan: list[KinematicState] = []
    kinematic_pick_plan = get_kinematic_plan_to_pick_object(
        kinematic_state,
        sim.robot,
        obj_id,
        surface_id,
        collision_ids,
        grasp_generator=grasp_generator,
        max_motion_planning_time=max_motion_planning_time,
        max_motion_planning_candidates=max_motion_planning_candidates,
        max_smoothing_iters_per_step=1,
        postgrasp_translation_magnitude=1e-2,
    )
    if kinematic_pick_plan is None:
        return None
    kinematic_plan.extend(kinematic_pick_plan)
    return get_pybullet_action_plan_from_kinematic_plan(kinematic_plan)


def get_target_base_pose(
    state: PyBulletState,
    object_name: str,
    sim: PyBulletEnv,
) -> Pose:
    """Get a base pose for the robot to move next to an object."""
    sim.set_state(state)
    object_id = sim.get_object_id_from_name(object_name)

    if object_name == "shelf":
        return sim.scene_spec.robot_base_pose  # initial base pose
    if object_name == "bed":
        return Pose((1.0, 0.2, 0.0))
    table_base_x = sim.scene_spec.robot_base_pose.position[0]
    table_base_y = sim.scene_spec.robot_base_pose.position[1] - 0.1
    table_base_z = sim.scene_spec.robot_base_pose.position[2]
    if object_name == "table":
        return Pose(
            (
                table_base_x,
                table_base_y,
                table_base_z,
            ),
            sim.scene_spec.robot_base_pose.orientation,
        )
    if object_name.startswith("side-table"):
        table_y = sim.scene_spec.table_pose.position[1]
        side_table_idx = int(object_name[len("side-table-")])
        side_table_y = sim.scene_spec.side_table_poses[side_table_idx].position[1]
        dy = side_table_y - table_y
        return Pose(
            (
                table_base_x,
                table_base_y + dy,
                table_base_z,
            ),
            sim.scene_spec.robot_base_pose.orientation,
        )
    if object_name in ["duster"] + sim.book_descriptions:
        surface_id = sim.get_surface_that_object_is_on(object_id)
        surface_name = sim.get_name_from_object_id(surface_id)
        return get_target_base_pose(state, surface_name, sim)

    raise NotImplementedError


def get_plan_to_move_next_to_object(
    state: PyBulletState,
    object_name: str,
    sim: PyBulletEnv,
    seed: int = 0,
) -> list[PyBulletAction]:
    """Get a plan to move next to a given object."""
    target_base_pose = get_target_base_pose(state, object_name, sim)
    return get_plan_to_move_to_pose(state, target_base_pose, sim, seed)


def get_plan_to_move_to_pose(
    state: PyBulletState,
    target_base_pose: Pose,
    sim: PyBulletEnv,
    seed: int = 0,
) -> list[PyBulletAction]:
    """Get a plan to move next to a given object."""
    sim.set_state(state)
    kinematic_state = get_kinematic_state_from_pybullet_state(state, sim)
    collision_ids = sim.get_collision_ids() - set(kinematic_state.attachments)
    current_base_pose = state.robot_base

    if kinematic_state.attachments:
        assert len(kinematic_state.attachments) == 1
        held_obj_id, held_obj_tf = next(iter(kinematic_state.attachments.items()))
    else:
        held_obj_id, held_obj_tf = None, None

    base_motion_plan = run_base_motion_planning(
        sim.robot,
        current_base_pose,
        target_base_pose,
        position_lower_bounds=sim.scene_spec.world_lower_bounds[:2],
        position_upper_bounds=sim.scene_spec.world_upper_bounds[:2],
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
    max_motion_planning_candidates: int = 1,
    max_motion_planning_time: float = np.inf,
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
        max_candidate_plans=max_motion_planning_candidates,
        max_time=max_motion_planning_time,
        held_object=object_id,
        base_link_to_held_obj=kinematic_state.attachments[object_id],
    )
    assert robot_joint_plan is not None
    kinematic_plan: list[KinematicState] = []
    for robot_joints in robot_joint_plan:
        kinematic_plan.append(kinematic_state.copy_with(robot_joints=robot_joints))

    return get_pybullet_action_plan_from_kinematic_plan(kinematic_plan)


def get_plan_to_reverse_handover_object(
    state: PyBulletState,
    object_name: str,
    relative_grasp: Pose,
    sim: PyBulletEnv,
    max_motion_planning_candidates: int = 1,
    max_motion_planning_time: float = np.inf,
) -> list[PyBulletAction]:
    """Get a plan to grasp an object held by a person."""
    sim.set_state(state)
    object_id = sim.get_object_id_from_name(object_name)
    assert sim.current_human_held_object_id == object_id
    assert sim.current_held_object_id is None
    kinematic_state = get_kinematic_state_from_pybullet_state(state, sim)
    collision_ids = sim.get_collision_ids(ignore_current_collisions=True)

    # Disable post-grasp.
    postgrasp_translation = Pose((0.0, 0.0, 0.0))
    grasp_generator = iter([relative_grasp])
    kinematic_state = get_kinematic_state_from_pybullet_state(state, sim)
    kinematic_plan: list[KinematicState] = []
    kinematic_pick_plan = get_kinematic_plan_to_pick_object(
        kinematic_state,
        sim.robot,
        object_id,
        sim.human.robot_id,  # used for toggled collision checking
        collision_ids,
        grasp_generator=grasp_generator,
        max_motion_planning_time=max_motion_planning_time,
        max_motion_planning_candidates=max_motion_planning_candidates,
        max_smoothing_iters_per_step=1,
        postgrasp_translation=postgrasp_translation,
    )
    assert kinematic_pick_plan is not None
    kinematic_plan.extend(kinematic_pick_plan)
    return get_pybullet_action_plan_from_kinematic_plan(kinematic_plan)


def get_plan_to_place_object(
    state: PyBulletState,
    object_name: str,
    surface_name: str,
    placement_pose: Pose,
    sim: PyBulletEnv,
    seed: int = 0,
    max_motion_planning_candidates: int = 1,
    max_motion_planning_time: float = np.inf,
    surface_link_id: int = -1,
) -> list[PyBulletAction] | None:
    """Get a plan to place a held object on a given surface."""
    sim.set_state(state)
    object_id = sim.get_object_id_from_name(object_name)
    surface_id = sim.get_object_id_from_name(surface_name)
    collision_ids = sim.get_collision_ids()
    placement_generator = iter([placement_pose])
    kinematic_state = get_kinematic_state_from_pybullet_state(state, sim)
    held_link_id = sim.duster_head_link_id if object_name == "duster" else -1
    object_extents = sim.get_default_half_extents(object_id, held_link_id)
    kinematic_plan = get_kinematic_plan_to_place_object(
        kinematic_state,
        sim.robot,
        object_id,
        surface_id,
        collision_ids,
        placement_generator,
        surface_link_id=surface_link_id,
        preplace_translation_magnitude=(1.25 * object_extents[2]),
        max_motion_planning_time=max_motion_planning_time,
        max_motion_planning_candidates=max_motion_planning_candidates,
        birrt_num_iters=10,
        max_smoothing_iters_per_step=1,
        retract_after=True,
    )
    if kinematic_plan is None:
        return None
    kinematic_state = kinematic_plan[-1]
    kinematic_state.set_pybullet(sim.robot)
    # Motion plan back to home joint positions.
    robot_joint_plan = run_motion_planning(
        sim.robot,
        kinematic_state.robot_joints,
        sim.robot.home_joint_positions,
        collision_bodies=collision_ids - {object_id},
        seed=seed,
        physics_client_id=sim.physics_client_id,
    )
    if robot_joint_plan is None:
        return None
    for robot_joints in robot_joint_plan:
        kinematic_plan.append(kinematic_state.copy_with(robot_joints=robot_joints))
    kinematic_state = kinematic_plan[-1]
    kinematic_state.set_pybullet(sim.robot)
    return get_pybullet_action_plan_from_kinematic_plan(kinematic_plan)


def get_plan_to_wipe_surface(
    state: PyBulletState,
    duster_name: str,
    surface_name: str,
    grasp_robot_base_pose: Pose,
    wipe_robot_base_pose: Pose,
    wipe_robot_joint_state: JointPositions,
    wipe_direction_num_rotations: int,
    sim: PyBulletEnv,
    surface_link_id: int = -1,
    seed: int = 0,
    off_surface_padding: float = 1e-3,
    max_motion_planning_iters: int = 10,
    max_motion_planning_candidates: int = 1,
    max_motion_planning_time: float = np.inf,
) -> list[PyBulletAction] | None:
    """Assuming a surface is clear of objects and the robot hand is empty."""

    sim.set_state(state)
    kinematic_state = get_kinematic_state_from_pybullet_state(state, sim)
    duster_id = sim.get_object_id_from_name(duster_name)
    collision_ids = sim.get_collision_ids() - {duster_id}

    # Make a plan for the duster head.
    duster_head_plan = get_duster_head_frame_wiping_plan(
        state,
        duster_name,
        surface_name,
        wipe_direction_num_rotations,
        sim,
        surface_link_id=surface_link_id,
        off_surface_padding=off_surface_padding,
    )

    kinematic_plan: list[KinematicState] = []
    if state.held_object is None:
        # Make a plan to pick the duster.
        # First motion plan in SE2 to the robot base pose.
        base_motion_plan = run_base_motion_planning(
            sim.robot,
            state.robot_base,
            grasp_robot_base_pose,
            position_lower_bounds=sim.scene_spec.world_lower_bounds[:2],
            position_upper_bounds=sim.scene_spec.world_upper_bounds[:2],
            collision_bodies=collision_ids | {duster_id},
            seed=seed,
            physics_client_id=sim.physics_client_id,
            platform=sim.robot_stand_id,
        )
        if base_motion_plan is None:
            return None
        for base_pose in base_motion_plan:
            kinematic_plan.append(kinematic_state.copy_with(robot_base_pose=base_pose))
        kinematic_state = kinematic_plan[-1]
        kinematic_state.set_pybullet(sim.robot)
        # Very unfortunate workaround to deal with the fact that set_pybullet()
        # does not know about robot platform.
        assert kinematic_state.robot_base_pose is not None
        sim.set_robot_base(kinematic_state.robot_base_pose)

        # Now make plan to grasp.
        surface_id = sim.get_surface_that_object_is_on(duster_id)
        grasp_pose = sim.scene_spec.duster_grasp
        grasp_generator = iter([grasp_pose])
        kinematic_pick_plan = get_kinematic_plan_to_pick_object(
            kinematic_state,
            sim.robot,
            duster_id,
            surface_id,
            collision_ids | {duster_id},
            grasp_generator=grasp_generator,
            max_motion_planning_time=max_motion_planning_time,
            max_motion_planning_candidates=max_motion_planning_candidates,
            max_smoothing_iters_per_step=1,
            postgrasp_translation_magnitude=1e-3,
        )
        if kinematic_pick_plan is None:
            return None
        kinematic_plan.extend(kinematic_pick_plan)
        kinematic_state = kinematic_plan[-1]
        kinematic_state.set_pybullet(sim.robot)
        sim.current_held_object_id = duster_id
        sim.current_grasp_transform = grasp_pose.invert()

        # Motion plan back to home joint positions.
        kinematic_move_to_home_plan = get_kinematic_plan_to_move_arm_home(
            kinematic_state, sim, seed=seed
        )
        if kinematic_move_to_home_plan is None:
            return None
        kinematic_plan.extend(kinematic_move_to_home_plan)
        kinematic_state = kinematic_plan[-1]
        kinematic_state.set_pybullet(sim.robot)

    assert duster_id in kinematic_state.attachments

    # First motion plan in SE2 to the robot base pose.
    assert kinematic_state.robot_base_pose is not None
    base_motion_plan = run_base_motion_planning(
        sim.robot,
        kinematic_state.robot_base_pose,
        wipe_robot_base_pose,
        position_lower_bounds=sim.scene_spec.world_lower_bounds[:2],
        position_upper_bounds=sim.scene_spec.world_upper_bounds[:2],
        collision_bodies=collision_ids,
        seed=seed,
        physics_client_id=sim.physics_client_id,
        platform=sim.robot_stand_id,
        held_object=duster_id,
        base_link_to_held_obj=kinematic_state.attachments[duster_id],
    )
    if base_motion_plan is None:
        return None

    for base_pose in base_motion_plan:
        kinematic_plan.append(kinematic_state.copy_with(robot_base_pose=base_pose))
    kinematic_state = kinematic_plan[-1]
    kinematic_state.set_pybullet(sim.robot)
    # Very unfortunate workaround to deal with the fact that set_pybullet()
    # does not know about robot platform.
    assert kinematic_state.robot_base_pose is not None
    sim.set_robot_base(kinematic_state.robot_base_pose)

    # Now motion plan in joint space to the initial pre-wiping pose.
    robot_joint_plan = run_motion_planning(
        sim.robot,
        kinematic_state.robot_joints,
        wipe_robot_joint_state,
        collision_bodies=collision_ids,
        seed=seed,
        physics_client_id=sim.physics_client_id,
        held_object=duster_id,
        base_link_to_held_obj=kinematic_state.attachments[duster_id],
        hyperparameters=MotionPlanningHyperparameters(
            birrt_num_iters=max_motion_planning_iters
        ),
    )
    if robot_joint_plan is None:
        return None
    for robot_joints in robot_joint_plan:
        kinematic_plan.append(kinematic_state.copy_with(robot_joints=robot_joints))
    kinematic_state = kinematic_plan[-1]
    kinematic_state.set_pybullet(sim.robot)

    # The duster head should now be in position to start duster_head_plan.
    duster_head_pose = get_link_pose(
        sim.duster_id, sim.duster_head_link_id, sim.physics_client_id
    )

    # Get the transform between the duster head and the robot base.
    current_base_pose = sim.robot.get_base_pose()
    base_to_head = multiply_poses(current_base_pose.invert(), duster_head_pose)

    # Move the robot base in x, y space to do the wiping.
    for head_pose in duster_head_plan:
        target_base_pose = multiply_poses(head_pose, base_to_head.invert())
        kinematic_state = kinematic_state.copy_with(robot_base_pose=target_base_pose)
        kinematic_plan.append(kinematic_state)
        kinematic_state.set_pybullet(sim.robot)
        # Very unfortunate workaround to deal with the fact that set_pybullet()
        # does not know about robot platform.
        assert kinematic_state.robot_base_pose is not None
        sim.set_robot_base(kinematic_state.robot_base_pose)
        current_base_pose = target_base_pose
        # Check for collisions.
        if check_collisions_with_held_object(
            sim.robot,
            collision_ids,
            sim.physics_client_id,
            held_object=duster_id,
            base_link_to_held_obj=kinematic_state.attachments[duster_id],
            joint_state=sim.robot.get_joint_positions(),
        ):
            return None
        for collision_body in collision_ids:
            if check_body_collisions(
                sim.robot_stand_id,
                collision_body,
                sim.physics_client_id,
                perform_collision_detection=False,
            ):
                return None

    # Motion plan back to home joint positions.
    kinematic_state = kinematic_plan[-1]
    move_to_home_plan = get_kinematic_plan_to_move_arm_home(
        kinematic_state, sim, seed=seed
    )
    if move_to_home_plan is None:
        return None
    kinematic_plan.extend(move_to_home_plan)
    return get_pybullet_action_plan_from_kinematic_plan(kinematic_plan)


def get_kinematic_plan_to_move_arm_home(
    kinematic_state: KinematicState, sim: PyBulletEnv, seed: int = 0
) -> list[KinematicState] | None:
    """Motion plan back to home joint positions."""
    collision_ids = sim.get_collision_ids()
    if kinematic_state.attachments:
        assert len(kinematic_state.attachments) == 1
        held_obj_id, held_obj_tf = next(iter(kinematic_state.attachments.items()))
        collision_ids.discard(held_obj_id)
    else:
        held_obj_id, held_obj_tf = None, None
    kinematic_state.set_pybullet(sim.robot)
    robot_joint_plan = run_motion_planning(
        sim.robot,
        kinematic_state.robot_joints,
        sim.robot.home_joint_positions,
        collision_bodies=collision_ids,
        seed=seed,
        held_object=held_obj_id,
        base_link_to_held_obj=held_obj_tf,
        physics_client_id=sim.physics_client_id,
    )
    if robot_joint_plan is None:
        return None
    kinematic_plan: list[KinematicState] = []
    for robot_joints in robot_joint_plan:
        kinematic_plan.append(kinematic_state.copy_with(robot_joints=robot_joints))
    return kinematic_plan


def get_plan_to_move_arm_home(
    state: PyBulletState, sim: PyBulletEnv, seed: int = 0
) -> list[PyBulletAction] | None:
    """Motion plan back to home joint positions."""
    sim.set_state(state)
    kinematic_state = get_kinematic_state_from_pybullet_state(state, sim)
    kinematic_plan = get_kinematic_plan_to_move_arm_home(kinematic_state, sim, seed)
    if kinematic_plan is None:
        return None
    kinematic_state = kinematic_plan[-1]
    kinematic_state.set_pybullet(sim.robot)
    return get_pybullet_action_plan_from_kinematic_plan(kinematic_plan)


def get_duster_head_frame_wiping_plan(
    state: PyBulletState,
    duster_name: str,
    surface_name: str,
    wipe_direction_num_rotations: int,
    sim: PyBulletEnv,
    surface_link_id: int = -1,
    off_surface_padding: float = 1e-3,
) -> list[Pose]:
    """Interpolate between poses in the duster head frame space.

    Note that this function does not do collision checking.
    """
    sim.set_state(state)

    assert 0 <= wipe_direction_num_rotations < 4
    wipe_yaw = (np.pi / 2) * wipe_direction_num_rotations
    surface_id = sim.get_object_id_from_name(surface_name)

    assert duster_name == "duster"
    duster_vert_half = sim.scene_spec.duster_head_forward_length
    duster_height_half = sim.scene_spec.duster_head_up_down_length
    duster_horiz_half = sim.scene_spec.duster_head_long_length

    # Create the starting poses from which we will wipe forward. The poses are
    # in the frame of the duster.
    aabb_min, aabb_max = p.getAABB(
        surface_id, linkIndex=surface_link_id, physicsClientId=sim.physics_client_id
    )
    surface_center = (
        (aabb_min[0] + aabb_max[0]) / 2,
        (aabb_min[1] + aabb_max[1]) / 2,
        aabb_max[2] + off_surface_padding,
    )
    # The wipe origin is a pose at the center of the surface facing in the
    # wipe direction (z axis pointing forward) with the duster shape offset.
    wipe_origin = Pose.from_rpy(surface_center, (-np.pi / 2, 0, wipe_yaw))

    if wipe_direction_num_rotations in {1, 3}:
        wipe_horiz = aabb_max[1] - aabb_min[1]
        wipe_vert = aabb_max[0] - aabb_min[0]
    else:
        assert wipe_direction_num_rotations in {0, 2}
        wipe_horiz = aabb_max[0] - aabb_min[0]
        wipe_vert = aabb_max[1] - aabb_min[1]

    prewipe_tf = Pose((-wipe_horiz / 2, 0, -wipe_vert / 2))
    # The prewipe origin is a pose at the bottom left hand corner of the surface
    # where wiping should start, but without the size of the duster considered.
    prewipe_origin = multiply_poses(wipe_origin, prewipe_tf)

    # Calculate the initial poses to start wiping.
    num_wipes = int(np.ceil(wipe_horiz / (2 * duster_horiz_half)))
    padding = duster_horiz_half / 4
    first_init_wipe_pose = multiply_poses(
        prewipe_origin,
        Pose(
            (
                duster_horiz_half + padding,
                -duster_height_half,
                duster_vert_half + padding,
            )
        ),
    )
    final_init_wipe_pose = multiply_poses(
        first_init_wipe_pose,
        Pose((wipe_horiz - 2 * duster_horiz_half - 2 * padding, 0, 0)),
    )
    init_wipe_poses = list(
        iter_between_poses(
            first_init_wipe_pose, final_init_wipe_pose, num_interp=(num_wipes - 1)
        )
    )

    # Calculate corresponding final poses to finish wiping.
    wipe_motion_tf = Pose((0, 0, wipe_vert - 2 * duster_vert_half - 2 * padding))
    terminal_wipe_poses = [
        multiply_poses(pose, wipe_motion_tf) for pose in init_wipe_poses
    ]

    # Join together the waypoints and then interpolate between them.
    wipe_pose_waypoints = [
        val for pair in zip(init_wipe_poses, terminal_wipe_poses) for val in pair
    ]

    # Add one final waypoint at the bottom center of the surface to make the
    # next motion planning easier.
    finish_pose = list(
        iter_between_poses(init_wipe_poses[0], init_wipe_poses[-1], num_interp=2)
    )[1]
    wipe_pose_waypoints.append(finish_pose)

    final_poses: list[Pose] = [wipe_pose_waypoints[0]]
    for waypoint1, waypoint2 in zip(
        wipe_pose_waypoints[:-1], wipe_pose_waypoints[1:], strict=True
    ):
        final_poses.extend(
            iter_between_poses(
                waypoint1,
                waypoint2,
                include_start=False,
            )
        )

    return final_poses
