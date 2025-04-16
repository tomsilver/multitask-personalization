"""CSP elements for the feeding environment."""

import logging
from pathlib import Path
from typing import Any, Collection

import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, set_pose
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    check_body_collisions,
    inverse_kinematics,
    set_robot_joints_with_held_object,
)
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot

from multitask_personalization.csp_generation import CSPGenerator
from multitask_personalization.envs.feeding.feeding_env import FeedingEnv
from multitask_personalization.envs.feeding.feeding_scene_spec import FeedingSceneSpec
from multitask_personalization.envs.feeding.feeding_structs import (
    CloseGripper,
    FeedingAction,
    FeedingState,
    GraspTool,
    MovePlate,
    MoveToEEPose,
    MoveToJointPositions,
    MoveToLastJointPositionswithEEPose,
    UngraspTool,
    WaitForUserInput,
)
from multitask_personalization.structs import (
    CSP,
    CSPConstraint,
    CSPCost,
    CSPPolicy,
    CSPSampler,
    CSPVariable,
    FunctionalCSPConstraint,
    FunctionalCSPSampler,
)
from multitask_personalization.utils import Threshold1DModel


class _FeedingCSPPolicy(CSPPolicy[FeedingState, FeedingAction]):

    def __init__(
        self, sim: FeedingEnv, csp_variables: Collection[CSPVariable], seed: int = 0
    ) -> None:
        super().__init__(csp_variables=csp_variables, seed=seed)
        self._sim = sim
        self._current_plan: list[FeedingAction] = []
        self._terminated = False

    def _get_plan(self, obs: FeedingState) -> list[FeedingAction] | None:
        if obs.user_request == "food":
            return self._get_food_plan(obs)
        if obs.user_request == "drink":
            return self._get_drink_plan(obs)
        if obs.user_request == "prepare":
            return self._get_prepare_plan(obs)
        raise NotImplementedError()

    def _get_food_plan(self, obs: FeedingState) -> list[FeedingAction] | None:
        scene_spec = self._sim.scene_spec

        current_plate_pose = obs.plate_pose
        new_plate_position = self._get_value("plate_position")
        new_plate_pose = _plate_position_to_pose(new_plate_position, current_plate_pose)

        plate_before_transfer_pose = _transform_pose_relative_to_plate(
            "before_transfer_pose", new_plate_pose, self._sim.scene_spec
        )

        plate_before_transfer_pos = _transform_joints_relative_to_plate(
            "before_transfer_pos", new_plate_pose, self._sim.robot, self._sim.scene_spec
        )

        above_plate_pos = _transform_joints_relative_to_plate(
            "above_plate_pos", new_plate_pose, self._sim.robot, self._sim.scene_spec
        )

        move_plate_plan: list[FeedingAction] = [MovePlate(new_plate_pose)]

        pick_utensil_plan: list[FeedingAction] = [
            MoveToJointPositions(scene_spec.retract_pos),
            CloseGripper(),
            MoveToJointPositions(scene_spec.utensil_above_mount_pos),
            MoveToEEPose(scene_spec.utensil_inside_mount),
            GraspTool("utensil"),
            MoveToEEPose(scene_spec.utensil_outside_mount),
            MoveToEEPose(scene_spec.utensil_outside_above_mount),
            MoveToJointPositions(plate_before_transfer_pos),
        ]

        acquire_bite_plan: list[FeedingAction] = [
            MoveToJointPositions(above_plate_pos),
        ]

        ready_for_transfer = [WaitForUserInput("ready for transfer?")]

        transfer_bite_plan: list[FeedingAction] = [
            MoveToJointPositions(plate_before_transfer_pos),
            MoveToEEPose(plate_before_transfer_pose),
            MoveToEEPose(scene_spec.outside_mouth_transfer_pose),
            MoveToEEPose(plate_before_transfer_pose),
        ]

        stow_utensil_plan: list[FeedingAction] = [
            MoveToJointPositions(scene_spec.utensil_outside_above_mount_pos),
            MoveToEEPose(scene_spec.utensil_outside_mount),
            MoveToEEPose(scene_spec.utensil_inside_mount),
            UngraspTool(),
            MoveToEEPose(scene_spec.utensil_above_mount),
            MoveToJointPositions(scene_spec.retract_pos),
        ]

        finish = [WaitForUserInput("done")]

        plan = (
            move_plate_plan
            + pick_utensil_plan
            + acquire_bite_plan
            + ready_for_transfer
            + transfer_bite_plan
            + stow_utensil_plan
            + finish
        )

        return plan

    def _get_drink_plan(self, obs: FeedingState) -> list[FeedingAction] | None:
        scene_spec = self._sim.scene_spec

        drink_staging_pos = _transform_joints_relative_to_drink(
            "drink_staging_pos", obs.drink_pose, self._sim.robot, scene_spec
        )
        drink_pre_grasp_pose = _transform_pose_relative_to_drink(
            "drink_default_pre_grasp_pose", obs.drink_pose, scene_spec
        )
        drink_inside_bottom_pose = _transform_pose_relative_to_drink(
            "drink_default_inside_bottom_pose", obs.drink_pose, scene_spec
        )
        drink_inside_top_pose = _transform_pose_relative_to_drink(
            "drink_default_inside_top_pose", obs.drink_pose, scene_spec
        )
        drink_post_grasp_pose = _transform_pose_relative_to_drink(
            "drink_default_post_grasp_pose", obs.drink_pose, scene_spec
        )
        drink_before_transfer_pos = _transform_joints_relative_to_drink(
            "drink_before_transfer_pos", obs.drink_pose, self._sim.robot, scene_spec
        )
        drink_before_transfer_pose = _transform_pose_relative_to_drink(
            "drink_before_transfer_pose", obs.drink_pose, scene_spec
        )

        pick_drink_plan: list[FeedingAction] = [
            MoveToJointPositions(scene_spec.retract_pos),
            CloseGripper(),
            MoveToJointPositions(scene_spec.drink_gaze_pos),
            MoveToJointPositions(drink_staging_pos),
            MoveToEEPose(drink_pre_grasp_pose),
            MoveToEEPose(drink_inside_bottom_pose),
            MoveToEEPose(drink_inside_top_pose),
            GraspTool("drink"),
            MoveToEEPose(drink_post_grasp_pose),
            MoveToJointPositions(drink_before_transfer_pos),
        ]

        ready_for_transfer = [WaitForUserInput("ready for transfer?")]

        transfer_drink_plan: list[FeedingAction] = [
            MoveToEEPose(drink_before_transfer_pose),
            MoveToEEPose(scene_spec.outside_mouth_transfer_pose),
            MoveToEEPose(drink_before_transfer_pose),
        ]

        stow_drink_plan: list[FeedingAction] = [
            MoveToLastJointPositionswithEEPose(drink_post_grasp_pose),
            MoveToLastJointPositionswithEEPose(drink_inside_top_pose),
            UngraspTool(),
            MoveToLastJointPositionswithEEPose(drink_inside_bottom_pose),
            MoveToLastJointPositionswithEEPose(drink_pre_grasp_pose),
            MoveToJointPositions(scene_spec.retract_pos),
        ]

        finish = [WaitForUserInput("done")]

        plan = (
            pick_drink_plan
            + ready_for_transfer
            + transfer_drink_plan
            + stow_drink_plan
            + finish
        )

        return plan

    def _get_prepare_plan(self, obs: FeedingState) -> list[FeedingAction] | None:
        scene_spec = self._sim.scene_spec

        current_drink_pose = obs.drink_pose
        new_drink_position = self._get_value("drink_position")
        new_drink_pose = _drink_position_to_pose(new_drink_position, current_drink_pose)

        current_drink_staging_pos = _transform_joints_relative_to_drink(
            "drink_staging_pos", current_drink_pose, self._sim.robot, scene_spec
        )
        current_drink_pre_grasp_pose = _transform_pose_relative_to_drink(
            "drink_default_pre_grasp_pose", current_drink_pose, scene_spec
        )
        current_drink_inside_bottom_pose = _transform_pose_relative_to_drink(
            "drink_default_inside_bottom_pose", current_drink_pose, scene_spec
        )
        current_drink_inside_top_pose = _transform_pose_relative_to_drink(
            "drink_default_inside_top_pose", current_drink_pose, scene_spec
        )
        current_drink_post_grasp_pose = _transform_pose_relative_to_drink(
            "drink_default_post_grasp_pose", current_drink_pose, scene_spec
        )

        new_drink_pre_grasp_pose = _transform_pose_relative_to_drink(
            "drink_default_pre_grasp_pose", new_drink_pose, scene_spec
        )
        new_drink_inside_bottom_pose = _transform_pose_relative_to_drink(
            "drink_default_inside_bottom_pose", new_drink_pose, scene_spec
        )
        new_drink_inside_top_pose = _transform_pose_relative_to_drink(
            "drink_default_inside_top_pose", new_drink_pose, scene_spec
        )
        new_drink_post_grasp_pose = _transform_pose_relative_to_drink(
            "drink_default_post_grasp_pose", new_drink_pose, scene_spec
        )

        current_plate_pose = obs.plate_pose
        new_plate_position = self._get_value("plate_position")
        new_plate_pose = _plate_position_to_pose(new_plate_position, current_plate_pose)
        move_plate_plan: list[FeedingAction] = [MovePlate(new_plate_pose)]

        pick_drink_plan: list[FeedingAction] = [
            MoveToJointPositions(scene_spec.retract_pos),
            CloseGripper(),
            MoveToJointPositions(scene_spec.drink_gaze_pos),
            MoveToJointPositions(current_drink_staging_pos),
            MoveToEEPose(current_drink_pre_grasp_pose),
            MoveToEEPose(current_drink_inside_bottom_pose),
            MoveToEEPose(current_drink_inside_top_pose),
            GraspTool("drink"),
            MoveToEEPose(current_drink_post_grasp_pose),
        ]

        stow_drink_plan: list[FeedingAction] = [
            MoveToEEPose(new_drink_post_grasp_pose),
            MoveToEEPose(new_drink_inside_top_pose),
            UngraspTool(),
            MoveToEEPose(new_drink_inside_bottom_pose),
            MoveToEEPose(new_drink_pre_grasp_pose),
            MoveToJointPositions(scene_spec.retract_pos),
        ]

        finish = [WaitForUserInput("done")]

        plan = move_plate_plan + pick_drink_plan + stow_drink_plan + finish

        return plan

    def reset(self, solution: dict[CSPVariable, Any]) -> None:
        super().reset(solution)
        self._current_plan = []
        self._terminated = False

    def step(self, obs: FeedingState) -> FeedingAction:
        if not self._current_plan:
            self._sim.set_state(obs)
            plan = self._get_plan(obs)
            assert plan is not None
            self._current_plan = plan
        action = self._current_plan.pop(0)
        if isinstance(action, MoveToLastJointPositionswithEEPose):
            joint_positions = self._sim.get_joint_positions_from_known_ee_pose(
                action.pose
            )
            action = MoveToJointPositions(joint_positions[:7])
        # Very hacky: we need to step the environment to trigger saving the
        # joint positions for the end effector pose, so we can use it above.
        # The original sin for this hack is that the policy is implemented
        # open-loop, but the need to save joint positions makes it closed-loop.
        self._sim.step(action)
        self._terminated = (
            isinstance(action, WaitForUserInput) and action.user_input == "done"
        )
        return action

    def check_termination(self, obs: FeedingState) -> bool:
        return self._terminated


class FeedingCSPGenerator(CSPGenerator[FeedingState, FeedingAction]):
    """Generate CSPs for the feeding environment."""

    def __init__(
        self, sim: FeedingEnv, occlusion_scale_model: Threshold1DModel, *args, **kwargs
    ) -> None:
        self._sim = sim
        self._occlusion_model = occlusion_scale_model
        super().__init__(*args, **kwargs)

    def save(self, model_dir: Path) -> None:
        print("WARNING: saving not yet implemented for FeedingCSPGenerator.")

    def load(self, model_dir: Path) -> None:
        print("WARNING: loading not yet implemented for FeedingCSPGenerator.")

    def _generate_variables(
        self,
        obs: FeedingState,
    ) -> tuple[list[CSPVariable], dict[CSPVariable, Any]]:

        # XY position of the plate.
        plate_position_domain = Box(
            np.array([-np.inf, -np.inf]),
            np.array([np.inf, np.inf]),
            dtype=np.float32,
        )
        plate_position = CSPVariable("plate_position", plate_position_domain)
        init_plate_position = (obs.plate_pose.position[0], obs.plate_pose.position[1])

        # XY position of the drink.
        drink_position_domain = Box(
            np.array([-np.inf, -np.inf]),
            np.array([np.inf, np.inf]),
            dtype=np.float32,
        )
        drink_position = CSPVariable("drink_position", drink_position_domain)
        init_drink_position = (obs.drink_pose.position[0], obs.drink_pose.position[1])

        return [plate_position, drink_position], {
            plate_position: init_plate_position,
            drink_position: init_drink_position,
        }

    def _generate_personal_constraints(
        self,
        obs: FeedingState,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:

        constraints: list[CSPConstraint] = []
        plate_position, drink_position = variables

        # NOTE: we are currently just using the MLE occlusion scale, rather than
        # using the full distribution. That means that "ours" will be equivalent
        # to "exploit_only". This is because we're not really running full
        # experiments in this environment.

        # TODO change back!!!!!!!!
        occlusion_scale = (
            1.0 - (self._occlusion_model.post_max + self._occlusion_model.post_min) / 2
        )
        # occlusion_scale = 0.999
        self._sim.set_occlusion_scale(occlusion_scale)
        logging.info(f"Set sim occlusion scale to {occlusion_scale:.3f}")

        def _user_view_unoccluded_by_utensil(
            plate_position: NDArray[np.float32],
        ) -> bool:
            self._sim.set_state(obs)
            new_plate_pose = _plate_position_to_pose(plate_position, obs.plate_pose)
            field_name = "above_plate_pos"
            try:
                robot_joints = _transform_joints_relative_to_plate(
                    field_name,
                    new_plate_pose,
                    self._sim.robot,
                    self._sim.scene_spec,
                    arm_joints_only=False,
                )
            except InverseKinematicsError:
                print("WARNING: IK failed within _user_view_unoccluded_by_utensil()")
                # from pybullet_helpers.gui import visualize_pose
                # visualize_pose(new_plate_pose, self._sim.physics_client_id)
                return False
            held_object_id = self._sim.get_object_id_from_name("utensil")
            held_object_tf = self._sim.scene_spec.utensil_held_object_tf
            set_robot_joints_with_held_object(
                self._sim.robot,
                self._sim.physics_client_id,
                held_object_id,
                held_object_tf,
                robot_joints,
            )
            self._sim.robot.set_finger_state(
                self._sim.scene_spec.tool_grasp_fingers_value
            )
            return not self._sim.robot_in_occlusion()

        user_view_unoccluded_by_utensil_constraint = FunctionalCSPConstraint(
            "user_view_unoccluded_by_utensil",
            [plate_position],
            _user_view_unoccluded_by_utensil,
        )

        if obs.user_request not in ("drink", "prepare-drink-only"):
            constraints.append(user_view_unoccluded_by_utensil_constraint)

        def _user_view_unoccluded_by_drink(
            drink_position: NDArray[np.float32],
        ) -> bool:
            self._sim.set_state(obs)
            new_drink_pose = _drink_position_to_pose(drink_position, obs.drink_pose)
            drink_post_grasp_pose = _transform_pose_relative_to_drink(
                "drink_default_post_grasp_pose", new_drink_pose, self._sim.scene_spec
            )
            # from pybullet_helpers.gui import visualize_pose
            # visualize_pose(new_drink_pose, self._sim.physics_client_id)
            try:
                robot_joints = inverse_kinematics(
                    self._sim.robot, drink_post_grasp_pose
                )
            except InverseKinematicsError:
                print("WARNING: IK failed within _user_view_unoccluded_by_drink()")
                return False
            held_object_id = self._sim.get_object_id_from_name("drink")
            held_object_tf = self._sim.scene_spec.drink_held_object_tf
            set_robot_joints_with_held_object(
                self._sim.robot,
                self._sim.physics_client_id,
                held_object_id,
                held_object_tf,
                robot_joints,
            )
            self._sim.robot.set_finger_state(
                self._sim.scene_spec.tool_grasp_fingers_value
            )
            return not self._sim.robot_in_occlusion()

        user_view_unoccluded_by_drink_constraint = FunctionalCSPConstraint(
            "user_view_unoccluded_by_drink",
            [drink_position],
            _user_view_unoccluded_by_drink,
        )

        if obs.user_request != "food":
            constraints.append(user_view_unoccluded_by_drink_constraint)

        return constraints

    def _generate_nonpersonal_constraints(
        self,
        obs: FeedingState,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:

        constraints: list[CSPConstraint] = []

        plate_position, drink_position = variables

        # # The plate position must be valid w.r.t. IK.
        # def _plate_position_is_kinematically_valid(
        #     plate_position: NDArray[np.float32],
        # ) -> bool:
        #     new_plate_pose = _plate_position_to_pose(plate_position, obs.plate_pose)
        #     for field_name in ["before_transfer_pos", "above_plate_pos"]:
        #         try:
        #             _transform_joints_relative_to_plate(
        #                 field_name,
        #                 new_plate_pose,
        #                 self._sim.robot,
        #                 self._sim.scene_spec,
        #             )
        #         except InverseKinematicsError:
        #             return False
        #     return True

        # plate_position_kinematically_valid_constraint = FunctionalCSPConstraint(
        #     "plate_position_kinematically_valid",
        #     [plate_position],
        #     _plate_position_is_kinematically_valid,
        # )

        # if obs.user_request != "drink":
        #     constraints.append(plate_position_kinematically_valid_constraint)

        # The plate and drink cannot be in collision.
        def _plate_drink_collision_free(
            plate_position: NDArray[np.float32],
            drink_position: NDArray[np.float32],
        ) -> bool:
            new_plate_pose = _plate_position_to_pose(plate_position, obs.plate_pose)
            new_drink_pose = _drink_position_to_pose(drink_position, obs.drink_pose)
            set_pose(self._sim.plate_id, new_plate_pose, self._sim.physics_client_id)
            set_pose(self._sim.drink_id, new_drink_pose, self._sim.physics_client_id)
            return not check_body_collisions(
                self._sim.plate_id, self._sim.drink_id, self._sim.physics_client_id
            )

        plate_drink_collision_free_constraint = FunctionalCSPConstraint(
            "plate_drink_collision_free",
            [plate_position, drink_position],
            _plate_drink_collision_free,
        )

        # the plate cannot be too far from the robot base.
        def _plate_position_reachable(
            plate_position: NDArray[np.float32],
        ) -> bool:
            new_plate_pose = _plate_position_to_pose(plate_position, obs.plate_pose)
            plate_pos = new_plate_pose.position[:2]
            print(f"plate is at a distance of {np.linalg.norm(plate_pos)}")
            return np.linalg.norm(plate_pos) < 0.8 and plate_pos[0] > 0.5 # Not too near user
        
        plate_position_reachable_constraint = FunctionalCSPConstraint(
            "plate_position_reachable",
            [plate_position],
            _plate_position_reachable,
        )

        # the drink cannot be too far from the robot base.
        def _drink_position_reachable(
            drink_position: NDArray[np.float32],
        ) -> bool:
            new_drink_pose = _drink_position_to_pose(drink_position, obs.drink_pose)
            drink_pos = new_drink_pose.position[:2]
            print(f"drink is at a distance of {np.linalg.norm(drink_pos)}")
            return np.linalg.norm(drink_pos) < 0.8 and drink_pos[0] > 0.5 # Not too near user
        
        drink_position_reachable_constraint = FunctionalCSPConstraint(
            "drink_position_reachable",
            [drink_position],
            _drink_position_reachable,
        )

        if obs.user_request != "drink":
            constraints.append(plate_drink_collision_free_constraint)
            # constraints.append(plate_position_reachable_constraint)
            # constraints.append(drink_position_reachable_constraint)
        else:
            constraints.append(drink_position_reachable_constraint)

        return constraints

    def _generate_exploit_cost(
        self,
        obs: FeedingState,
        variables: list[CSPVariable],
    ) -> CSPCost | None:
        return None

    def _generate_samplers(
        self,
        obs: FeedingState,
        csp: CSP,
    ) -> list[CSPSampler]:

        # Sample plate positions.
        plate_position, drink_position = csp.variables

        samplers = []

        def _sample_plate_position(
            _: dict[CSPVariable, Any], rng: np.random.Generator
        ) -> dict[CSPVariable, Any]:
            radius = rng.uniform(0, self._sim.scene_spec.table_radius - self._sim.scene_spec.plate_radius)
            angle = rng.uniform(0, 2 * np.pi)
            origin = self._sim.scene_spec.table_pose.position[:2]
            new_pos = np.array(
                [
                    origin[0] + radius * np.cos(angle),
                    origin[1] + radius * np.sin(angle),
                ]
            ).astype(np.float32)
            return {plate_position: new_pos}

        plate_position_sampler = FunctionalCSPSampler(
            _sample_plate_position, csp, {plate_position}
        )
        if obs.user_request != "prepare-drink-only":
            samplers.append(plate_position_sampler)

        def _sample_drink_position(
            _: dict[CSPVariable, Any], rng: np.random.Generator
        ) -> dict[CSPVariable, Any]:
            radius = rng.uniform(0, self._sim.scene_spec.table_radius - self._sim.scene_spec.drink_radius)
            angle = rng.uniform(0, 2 * np.pi)
            origin = self._sim.scene_spec.table_pose.position[:2]
            new_pos = np.array(
                [
                    origin[0] + radius * np.cos(angle),
                    origin[1] + radius * np.sin(angle),
                ]
            ).astype(np.float32)
            return {drink_position: new_pos}

        drink_position_sampler = FunctionalCSPSampler(
            _sample_drink_position, csp, {drink_position}
        )
        samplers.append(drink_position_sampler)

        return samplers

    def _generate_policy(
        self,
        obs: FeedingState,
        csp_variables: Collection[CSPVariable],
    ) -> CSPPolicy:
        return _FeedingCSPPolicy(self._sim, csp_variables, self._seed)

    def observe_transition(
        self,
        obs: FeedingState,
        act: FeedingAction,
        next_obs: FeedingState,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        above_plate_pos = _transform_joints_relative_to_plate(
            "above_plate_pos", obs.plate_pose, self._sim.robot, self._sim.scene_spec
        )
        # When we do real experiments, we will decide whether to take natural
        # language here and detect whether it's feedback about occlusion, or
        # to keep it simple we might just keep it binary (occluding or not).
        if next_obs.user_feedback == "You're blocking my view!":
            label = True
        # Positive examples are collected when the robot is at the above plate
        # position and no negative feedback is given.
        elif isinstance(act, MoveToJointPositions) and np.allclose(
            act.joint_positions, above_plate_pos
        ):
            label = False
        else:
            return
        self._sim.set_state(next_obs)
        occlusion_score = self._sim.get_occlusion_score()
        self._occlusion_model.fit_incremental([occlusion_score], [label])
        print(f"Updated occlusion model with {occlusion_score}, {label}")
        print(f"New params: {self._occlusion_model.get_summary()}")


def _plate_position_to_pose(
    plate_position: NDArray[np.float32], default_pose: Pose
) -> Pose:
    return Pose(
        (
            plate_position[0],
            plate_position[1],
            default_pose.position[2],
        ),
        default_pose.orientation,
    )


def _drink_position_to_pose(
    drink_position: NDArray[np.float32], default_pose: Pose
) -> Pose:
    return Pose(
        (
            drink_position[0],
            drink_position[1],
            default_pose.position[2],
        ),
        default_pose.orientation,
    )


def _transform_joints_relative_to_plate(
    scene_spec_field: str,
    plate_pose: Pose,
    sim_robot: FingeredSingleArmPyBulletRobot,
    scene_spec: FeedingSceneSpec,
    arm_joints_only: bool = True,
) -> JointPositions:
    return _transform_joints_relative_to_default(
        scene_spec_field,
        "plate_default_pose",
        plate_pose,
        sim_robot,
        scene_spec,
        arm_joints_only=arm_joints_only,
    )


def _transform_joints_relative_to_drink(
    scene_spec_field: str,
    drink_pose: Pose,
    sim_robot: FingeredSingleArmPyBulletRobot,
    scene_spec: FeedingSceneSpec,
    arm_joints_only: bool = True,
) -> JointPositions:
    return _transform_joints_relative_to_default(
        scene_spec_field,
        "drink_default_pose",
        drink_pose,
        sim_robot,
        scene_spec,
        arm_joints_only=arm_joints_only,
    )


def _transform_joints_relative_to_default(
    scene_spec_field: str,
    default_scene_field: str,
    pose: Pose,
    sim_robot: FingeredSingleArmPyBulletRobot,
    scene_spec: FeedingSceneSpec,
    arm_joints_only: bool = True,
) -> JointPositions:
    default_positions = getattr(scene_spec, scene_spec_field)
    world_to_default: Pose = getattr(scene_spec, default_scene_field)
    full_joints = sim_robot.get_joint_positions()
    num_dof = len(default_positions)
    full_joints[:num_dof] = default_positions
    sim_robot.set_joints(full_joints)
    world_to_ee = sim_robot.get_end_effector_pose()
    plate_to_ee = world_to_default.invert().multiply(world_to_ee)
    new_ee = pose.multiply(plate_to_ee)
    new_full_joints = inverse_kinematics(sim_robot, new_ee)
    if arm_joints_only:
        return new_full_joints[:num_dof]
    return new_full_joints


def _transform_pose_relative_to_plate(
    scene_spec_field: str, plate_pose: Pose, scene_spec: FeedingSceneSpec
) -> Pose:
    return _transform_pose_relative_to_default(
        scene_spec_field, "plate_default_pose", plate_pose, scene_spec
    )


def _transform_pose_relative_to_drink(
    scene_spec_field: str, drink_pose: Pose, scene_spec: FeedingSceneSpec
) -> Pose:
    return _transform_pose_relative_to_default(
        scene_spec_field, "drink_default_pose", drink_pose, scene_spec
    )


def _transform_pose_relative_to_default(
    pose_scene_spec_field: str,
    default_scene_field: str,
    pose: Pose,
    scene_spec: FeedingSceneSpec,
) -> Pose:
    world_to_pose: Pose = getattr(scene_spec, pose_scene_spec_field)
    world_to_default: Pose = getattr(scene_spec, default_scene_field)
    plate_to_pose = world_to_default.invert().multiply(world_to_pose)
    new_pose = pose.multiply(plate_to_pose)
    return new_pose
