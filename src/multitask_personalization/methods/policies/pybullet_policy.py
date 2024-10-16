"""A domain-specific parameterized policy for pybullet."""

from typing import Iterator

import numpy as np
from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
)
from pybullet_helpers.link import get_link_pose
from pybullet_helpers.manipulation import get_kinematic_plan_to_pick_object
from pybullet_helpers.math_utils import get_poses_facing_line
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    run_smooth_motion_planning_to_pose,
)
from pybullet_helpers.states import KinematicState

from multitask_personalization.envs.pybullet.pybullet_scene_description import (
    PyBulletSceneDescription,
)
from multitask_personalization.envs.pybullet.pybullet_sim import (
    PyBulletSimulator,
)
from multitask_personalization.envs.pybullet.pybullet_structs import (
    _GripperAction,
    _PyBulletAction,
    _PyBulletState,
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
        scene_description: PyBulletSceneDescription,
        seed: int = 0,
        max_motion_planning_time: float = 1.0,
    ) -> None:
        super().__init__()
        self._seed = seed
        self._max_motion_planning_time = max_motion_planning_time
        self._rng = np.random.default_rng(seed)
        # Create a simulator for planning.
        self._sim = PyBulletSimulator(scene_description, use_gui=False)
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
            self._sim.table_id: self._sim.scene_description.table_pose,
        }
        attachments: dict[int, Pose] = {}
        if state.grasp_transform:
            attachments[self._sim.object_id] = state.grasp_transform
        return KinematicState(robot_joints, object_poses, attachments)

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

        collision_ids = {self._sim.table_id, self._sim.human.body}

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
            self._sim.object_id,
            self._sim.table_id,
            collision_ids,
            grasp_generator=_grasp_generator(),
        )
        assert kinematic_plan is not None

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
        joint_delta = np.subtract(next_state.robot_joints, state.robot_joints)
        delta = joint_delta[:7]
        actions: list[_PyBulletAction] = [(0, delta.tolist())]
        if next_state.attachments and not state.attachments:
            actions.append((1, _GripperAction.CLOSE))
        elif state.attachments and not next_state.attachments:
            actions.append((1, _GripperAction.OPEN))
        return actions
