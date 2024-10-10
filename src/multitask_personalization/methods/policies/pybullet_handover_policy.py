"""A domain-specific parameterized policy for pybullet handover."""

import numpy as np
from pybullet_helpers.geometry import Pose, interpolate_poses, multiply_poses
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
)
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.link import get_link_pose
from pybullet_helpers.math_utils import get_poses_facing_line
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)

from multitask_personalization.envs.pybullet_handover import (
    PyBulletHandoverSceneDescription,
    PyBulletHandoverSimulator,
    _GripperAction,
    _HandoverAction,
    _HandoverState,
)
from multitask_personalization.methods.policies.parameterized_policy import (
    ParameterizedPolicy,
)
from multitask_personalization.utils import sample_spherical


class PyBulletHandoverParameterizedPolicy(
    ParameterizedPolicy[_HandoverState, _HandoverAction, float]
):
    """A domain-specific parameterized policy for pybullet handover.

    Parameters define the range of motion model. For now, we have a very simple
    ROM model that only has one parameter: the radius of a sphere around the
    person's hand.
    """

    def __init__(
        self,
        scene_description: PyBulletHandoverSceneDescription,
        seed: int = 0,
        max_motion_planning_time: float = 1.0,
    ) -> None:
        super().__init__()
        self._seed = seed
        self._max_motion_planning_time = max_motion_planning_time
        self._rng = np.random.default_rng(seed)
        # Create a simulator for planning.
        self._sim = PyBulletHandoverSimulator(scene_description, use_gui=False)
        self._joint_distance_fn = create_joint_distance_fn(self._sim.robot)
        # Store an action plan for the robot.
        self._plan: list[_HandoverAction] = []

    def reset(self, task_id: str, parameters: float) -> None:
        super().reset(task_id, parameters)
        self._plan = []

    def step(self, state: _HandoverState) -> _HandoverAction:
        assert self._current_parameters is not None
        if np.isinf(self._current_parameters):
            return (2, None)
        if not self._plan:
            # Sample a handover position on the surface of the current
            # estimated sphere. Repeatedly sample until a plan is found.
            while True:
                self._sim.set_state(state)
                handover_pose = self._sample_handover_pose(self._current_parameters)
                try:
                    inverse_kinematics(self._sim.robot, handover_pose)
                    break
                except InverseKinematicsError:
                    continue
            self._sim.set_state(state)
            self._plan = self._get_handover_plan(state, handover_pose)
        assert len(self._plan) > 0
        return self._plan.pop(0)

    def _sample_handover_pose(self, radius: float) -> Pose:
        # Get the sphere center from the simulator.
        center = get_link_pose(
            self._sim.human.body,
            self._sim.human.right_wrist,
            self._sim.physics_client_id,
        ).position
        position = sample_spherical(center, radius, self._rng)
        orientation = self._sim.robot.get_end_effector_pose().orientation
        return Pose(position, orientation)

    def _rollout_pybullet_helpers_plan(
        self, plan: list[JointPositions]
    ) -> list[_HandoverAction]:
        rollout = []
        assert plan is not None
        plan = remap_joint_position_plan_to_constant_distance(plan, self._sim.robot)
        for joint_state in plan:
            sim_state = self._sim.get_state()
            joint_delta = np.subtract(joint_state, sim_state.robot_joints)
            delta = joint_delta[:7]
            action = (0, delta.tolist())
            rollout.append(action)
            self._sim.step(action)
        return rollout

    def _get_handover_plan(
        self, state: _HandoverState, handover_pose: Pose
    ) -> list[_HandoverAction]:
        plan: list[_HandoverAction] = []

        # This should only happen in the case where the policy fails.
        if state.grasp_transform is not None:
            return [(2, None)]

        self._sim.set_state(state)
        object_pose = state.object_pose
        collision_ids = {self._sim.table_id, self._sim.human.body}

        # Sample grasp poses for the object.
        pybullet_helpers_plan: list[JointPositions] | None = None
        while pybullet_helpers_plan is None:
            self._sim.set_state(state)
            angle_offset = self._rng.uniform(-np.pi, np.pi)
            relative_pose = get_poses_facing_line(
                axis=(0.0, 0.0, 1.0),
                point_on_line=(0.0, 0.0, 0),
                radius=0.1,
                num_points=1,
                angle_offset=angle_offset,
            )[0]
            pregrasp_pose = multiply_poses(object_pose, relative_pose)
            pybullet_helpers_plan = run_smooth_motion_planning_to_pose(
                pregrasp_pose,
                self._sim.robot,
                collision_ids=collision_ids,
                end_effector_frame_to_plan_frame=Pose.identity(),
                seed=self._seed,
                max_time=self._max_motion_planning_time,
            )
        self._sim.set_state(state)
        plan.extend(self._rollout_pybullet_helpers_plan(pybullet_helpers_plan))
        state = self._sim.get_state()

        # Move forward to grasp.
        end_effector_pose = self._sim.robot.get_end_effector_pose()
        end_effector_path = list(
            interpolate_poses(
                end_effector_pose,
                Pose(
                    object_pose.position,
                    end_effector_pose.orientation,
                ),
                include_start=False,
            )
        )
        pregrasp_to_grasp_pybullet_helpers_plan = smoothly_follow_end_effector_path(
            self._sim.robot,
            end_effector_path,
            self._sim.robot.get_joint_positions(),
            collision_ids,
            self._joint_distance_fn,
            max_time=self._max_motion_planning_time,
            include_start=False,
        )
        assert pregrasp_to_grasp_pybullet_helpers_plan is not None
        self._sim.set_state(state)
        plan.extend(
            self._rollout_pybullet_helpers_plan(pregrasp_to_grasp_pybullet_helpers_plan)
        )
        state = self._sim.get_state()

        # Close the gripper.
        action = (1, _GripperAction.CLOSE)
        plan.append(action)
        self._sim.step(action)

        # Move up to remove contact with table.
        end_effector_pose = self._sim.robot.get_end_effector_pose()
        post_grasp_pose = Pose(
            (
                end_effector_pose.position[0],
                end_effector_pose.position[1],
                end_effector_pose.position[2] + 1e-2,
            ),
            end_effector_pose.orientation,
        )
        end_effector_path = list(
            interpolate_poses(
                end_effector_pose,
                post_grasp_pose,
                include_start=False,
            )
        )
        grasp_to_post_grasp_pybullet_helpers_plan = smoothly_follow_end_effector_path(
            self._sim.robot,
            end_effector_path,
            self._sim.robot.get_joint_positions(),
            collision_ids,
            self._joint_distance_fn,
            max_time=self._max_motion_planning_time,
            include_start=False,
            held_object=self._sim.object_id,
            base_link_to_held_obj=self._sim.current_grasp_transform,
        )
        assert grasp_to_post_grasp_pybullet_helpers_plan is not None
        self._sim.set_state(state)
        plan.extend(
            self._rollout_pybullet_helpers_plan(
                grasp_to_post_grasp_pybullet_helpers_plan
            )
        )
        state = self._sim.get_state()

        # Motion plan to the handover pose.
        handover_pybullet_helpers_plan = run_smooth_motion_planning_to_pose(
            handover_pose,
            self._sim.robot,
            collision_ids=collision_ids,
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=self._seed,
            max_time=self._max_motion_planning_time,
        )
        # This can happen if a handover position that is out of reach for the
        # robot is sampled.
        assert handover_pybullet_helpers_plan is not None
        self._sim.set_state(state)
        plan.extend(self._rollout_pybullet_helpers_plan(handover_pybullet_helpers_plan))

        return plan
