"""A domain-specific parameterized policy for pybullet."""

from typing import Iterator

import numpy as np
from pybullet_helpers.geometry import Pose, multiply_poses, get_pose, set_pose
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
)
from pybullet_helpers.link import get_link_pose
from pybullet_helpers.manipulation import get_kinematic_plan_to_pick_object, get_kinematic_plan_to_place_object
from pybullet_helpers.math_utils import get_poses_facing_line
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    run_smooth_motion_planning_to_pose,
    run_base_motion_planning_to_goal,
)
from pybullet_helpers.states import KinematicState

from multitask_personalization.envs.pybullet.pybullet_sim import (
    PyBulletSimulator,
)
from multitask_personalization.envs.pybullet.pybullet_structs import (
    _GripperAction,
    _PyBulletAction,
    _PyBulletState,
)
from multitask_personalization.envs.pybullet.pybullet_task_spec import (
    PyBulletTaskSpec,
)
from multitask_personalization.methods.policies.parameterized_policy import (
    ParameterizedPolicy,
)
from multitask_personalization.utils import sample_spherical


class PyBulletParameterizedPolicy(
    ParameterizedPolicy[_PyBulletState, _PyBulletAction, float]
):
    """A domain-specific parameterized policy for pybullet.

    Parameters define the range of motion model. For now, we have a very simple
    ROM model that only has one parameter: the radius of a sphere around the
    person's hand.
    """

    def __init__(
        self,
        task_spec: PyBulletTaskSpec,
        seed: int = 0,
        max_motion_planning_time: float = 1.0,
    ) -> None:
        super().__init__()
        self._seed = seed
        self._max_motion_planning_time = max_motion_planning_time
        self._rng = np.random.default_rng(seed)
        self._task_spec = task_spec
        # Create a simulator for planning.
        self._sim = PyBulletSimulator(task_spec, use_gui=True)
        self._joint_distance_fn = create_joint_distance_fn(self._sim.robot)
        # Store an action plan for the robot.
        self._plan: list[_PyBulletAction] = []

    def reset(self, task_id: str, parameters: float) -> None:
        super().reset(task_id, parameters)
        self._plan = []

    def step(self, state: _PyBulletState) -> _PyBulletAction:
        assert self._current_parameters is not None
        if np.isinf(self._current_parameters):
            return (2, None)
        if not self._plan:
            kinematic_state = self._pybullet_state_to_kinematic_state(state)
            # This should only happen in the case where the policy fails.
            if kinematic_state.attachments:
                return (2, None)
            kinematic_plan = self._get_kinematic_plan(kinematic_state)
            self._plan = self._kinematic_plan_to_pybullet_plan(kinematic_plan)
        assert len(self._plan) > 0
        return self._plan.pop(0)

    def _pybullet_state_to_kinematic_state(
        self, state: _PyBulletState
    ) -> KinematicState:
        robot_joints = state.robot_joints
        object_poses = {
            self._sim.object_id: state.object_pose,
            self._sim.book_id: state.book_pose,
            self._sim.table_id: self._task_spec.table_pose,
            self._sim.shelf_id: self._task_spec.shelf_pose,
            self._sim.tray_id: self._task_spec.tray_pose,
        }
        attachments: dict[int, Pose] = {}
        if state.grasp_transform:
            attachments[self._sim.object_id] = state.grasp_transform
        return KinematicState(robot_joints, object_poses, attachments, state.robot_base)

    def _sample_pybullet_pose(self, radius: float) -> Pose:
        # Get the sphere center from the simulator.
        center = get_link_pose(
            self._sim.human.body,
            self._sim.human.right_wrist,
            self._sim.physics_client_id,
        ).position
        position = sample_spherical(center, radius, self._rng)
        orientation = (
            0.8522037863731384,
            0.4745013415813446,
            -0.01094298530369997,
            0.22017613053321838,
        )
        pose = Pose(position, orientation)
        return pose

    def _get_kinematic_plan(
        self, initial_state: KinematicState
    ) -> list[KinematicState]:

        if self._task_spec.task_objective == "hand over cup":
            object_id = self._sim.object_id
            surface_id = self._sim.table_id
        elif self._task_spec.task_objective == "hand over book":
            object_id = self._sim.book_id
            surface_id = self._sim.shelf_id
        elif self._task_spec.task_objective == "place book on tray":
            object_id = self._sim.book_id
            surface_id = self._sim.shelf_id
        else:
            raise NotImplementedError

        collision_ids = {self._sim.table_id, self._sim.human.body, self._sim.shelf_id,
                         self._sim.tray_id, self._sim.side_table_id}

        def _grasp_generator() -> Iterator[Pose]:
            while True:
                angle_offset = self._rng.uniform(-np.pi, np.pi)
                relative_pose = get_poses_facing_line(
                    axis=(0.0, 0.0, 1.0),
                    point_on_line=(0.0, 0.0, 0),
                    radius=1e-3,
                    num_points=1,
                    angle_offset=angle_offset,
                )[0]
                yield relative_pose

        kinematic_plan = get_kinematic_plan_to_pick_object(
            initial_state,
            self._sim.robot,
            object_id,
            surface_id,
            collision_ids,
            grasp_generator=_grasp_generator(),
        )
        assert kinematic_plan is not None

        if self._task_spec.task_objective == "place book on tray":
            
            state = kinematic_plan[-1]
            state.set_pybullet(self._sim.robot)
            current_base_pose = self._sim.robot.get_base_pose()
            current_ee_pose = self._sim.robot.forward_kinematics(state.robot_joints)

            # Prepare to set platform after.
            world_to_base = self._sim.robot.get_base_pose()
            world_to_platform = get_pose(self._sim.robot_stand_id, self._sim.physics_client_id)
            base_to_platform = multiply_poses(world_to_base.invert(), world_to_platform)

            # Set up at target area for base position motion planning that
            # checks the position of the end effector and sees whether it is
            # close enough to the tray. Then run motion planning in SE2 for the
            # base only.
            ideal_pre_place_ee_pose = Pose(
                (self._task_spec.tray_pose.position[0] - 0.25,
                self._task_spec.tray_pose.position[1],
                self._task_spec.tray_pose.position[2] + 0.25),
                current_ee_pose.orientation)
            
            # TODO remove
            # import pybullet as p
            # from pybullet_helpers.gui import visualize_pose
            # while True:
            #     visualize_pose(ideal_pre_place_ee_pose, self._sim.physics_client_id)

            def _goal_check(base_pose: Pose) -> bool:
                self._sim.robot.set_base(base_pose)
                ee_pose = self._sim.robot.forward_kinematics(state.robot_joints)
                return np.linalg.norm(np.subtract(ideal_pre_place_ee_pose.position, ee_pose.position)) < 0.25 and np.linalg.norm(np.subtract(ideal_pre_place_ee_pose.orientation, ee_pose.orientation)) < 1.0

            base_motion_plan = run_base_motion_planning_to_goal(
                self._sim.robot,
                current_base_pose,
                _goal_check,
                position_lower_bounds=self._task_spec.world_lower_bounds[:2],
                position_upper_bounds=self._task_spec.world_upper_bounds[:2],
                collision_bodies=collision_ids,
                seed=self._seed,
                physics_client_id=self._sim.physics_client_id,
                platform=self._sim.robot_stand_id,
                held_object=object_id,
                base_link_to_held_obj=state.attachments[object_id],
            )

            assert base_motion_plan is not None

            # Extend the kinematic plan.
            for base_pose in base_motion_plan:
                kinematic_plan.append(state.copy_with(robot_base_pose=base_pose))

            # Also update the platform in sim.
            state = kinematic_plan[-1]
            state.set_pybullet(self._sim.robot)
            platform_pose = multiply_poses(state.robot_base_pose, base_to_platform)
            set_pose(self._sim.robot_stand_id, platform_pose, self._sim.physics_client_id)

            # Prepare to place.
            half_extents = self._task_spec.tray_half_extents
            object_radius = max(self._task_spec.book_half_extents[:2])
            object_length = 2 * self._task_spec.book_half_extents[2]
            placement_lb = (
                -half_extents[0] + object_radius,
                -half_extents[1] + object_radius,
                half_extents[2] + object_length / 2,
            )
            placement_ub = (
                half_extents[0] - object_radius,
                half_extents[1] - object_radius,
                half_extents[2] + object_length / 2,
            )

            def _placement_generator():
                # Sample on the surface of the table.
                while True:
                    yield Pose(tuple(self._rng.uniform(placement_lb, placement_ub)))
            
            placement_kinematic_plan = get_kinematic_plan_to_place_object(state, self._sim.robot, object_id,
                                               self._sim.tray_id, collision_ids,
                                               _placement_generator(), max_motion_planning_time=self._max_motion_planning_time)
            assert placement_kinematic_plan is not None
            kinematic_plan.extend(placement_kinematic_plan)

            import time
            for state in placement_kinematic_plan:
                state.set_pybullet(self._sim.robot)
                time.sleep(0.1)
            import ipdb; ipdb.set_trace()


            return kinematic_plan

        # Sample a reachable handover pose.
        handover_pose: Pose | None = None
        while True:
            assert self._current_parameters is not None
            candidate = self._sample_pybullet_pose(self._current_parameters)
            try:
                inverse_kinematics(self._sim.robot, candidate)
                handover_pose = candidate
                break
            except InverseKinematicsError:
                continue
        assert handover_pose is not None

        # Motion plan to hand over.
        state = kinematic_plan[-1]
        state.set_pybullet(self._sim.robot)
        robot_joint_plan = run_smooth_motion_planning_to_pose(
            handover_pose,
            self._sim.robot,
            collision_ids=collision_ids,
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=self._seed,
            max_time=self._max_motion_planning_time,
            held_object=object_id,
            base_link_to_held_obj=state.attachments[object_id],
        )
        assert robot_joint_plan is not None
        for robot_joints in robot_joint_plan:
            kinematic_plan.append(state.copy_with(robot_joints=robot_joints))

        return kinematic_plan

    def _kinematic_plan_to_pybullet_plan(
        self, kinematic_plan: list[KinematicState]
    ) -> list[_PyBulletAction]:
        actions: list[_PyBulletAction] = []
        for s0, s1 in zip(kinematic_plan[:-1], kinematic_plan[1:], strict=True):
            step_actions = self._kinematic_transition_to_actions(s0, s1)
            actions.extend(step_actions)
        return actions

    def _kinematic_transition_to_actions(
        self, state: KinematicState, next_state: KinematicState
    ) -> list[_PyBulletAction]:
        base_delta = (
            next_state.robot_base_pose.position[0] - state.robot_base_pose.position[0],
            next_state.robot_base_pose.position[1] - state.robot_base_pose.position[1],
            next_state.robot_base_pose.rpy[2] - state.robot_base_pose.rpy[2]
        )
        joint_delta = np.subtract(next_state.robot_joints, state.robot_joints)
        delta = list(base_delta) + list(joint_delta[:7])
        actions: list[_PyBulletAction] = [(0, delta)]
        if next_state.attachments and not state.attachments:
            actions.append((1, _GripperAction.CLOSE))
        elif state.attachments and not next_state.attachments:
            actions.append((1, _GripperAction.OPEN))
        return actions
