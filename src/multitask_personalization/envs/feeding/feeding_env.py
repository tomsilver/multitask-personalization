"""Assistive feeding environment in pybullet."""

from __future__ import annotations

import logging
import time
from typing import Any

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium.core import RenderFrame
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import (
    Pose,
    get_pose,
    iter_between_poses,
    multiply_poses,
    set_pose,
)
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.inverse_kinematics import (
    check_body_collisions,
    set_robot_joints_with_held_object,
)
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.link import get_relative_link_pose
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block, create_pybullet_cylinder
from tomsutils.spaces import FunctionalSpace

from multitask_personalization.envs.feeding.feeding_hidden_spec import (
    FeedingHiddenSceneSpec,
)
from multitask_personalization.envs.feeding.feeding_scene_spec import FeedingSceneSpec
from multitask_personalization.envs.feeding.feeding_structs import (
    CloseGripper,
    FeedingAction,
    FeedingState,
    GraspTool,
    MoveDrink,
    MovePlate,
    MoveToEEPose,
    MoveToJointPositions,
    UngraspTool,
    WaitForUserInput,
)
from multitask_personalization.envs.feeding.feeding_utils import cartesian_control_step
from multitask_personalization.envs.pybullet.pybullet_utils import (
    BANISH_POSE,
)


class FeedingEnv(gym.Env[FeedingState, FeedingAction]):
    """An assistive feeding environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 2}

    def __init__(
        self,
        scene_spec: FeedingSceneSpec,
        hidden_spec: FeedingHiddenSceneSpec | None = None,
        use_gui: bool = False,
        seed: int = 0,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self.scene_spec = scene_spec
        self._hidden_spec = hidden_spec
        self.render_mode = "rgb_array"
        self.action_space = FunctionalSpace(
            contains_fn=lambda action: isinstance(action, FeedingAction)
        )
        self._use_gui = use_gui

        # Create the PyBullet client.
        if use_gui:
            camera_kwargs = self.scene_spec.get_camera_kwargs()
            self.physics_client_id = create_gui_connection(**camera_kwargs)
        else:
            self.physics_client_id = p.connect(p.DIRECT)

        # Create floor.
        self.floor_id = p.loadURDF(
            str(self.scene_spec.floor_urdf),
            self.scene_spec.floor_position,
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )

        # Create robot.
        robot = create_pybullet_robot(
            self.scene_spec.robot_name,
            self.physics_client_id,
            base_pose=self.scene_spec.robot_base_pose,
            control_mode="reset",
            home_joint_positions=self.scene_spec.initial_joints,
            custom_urdf_path=self.scene_spec.robot_urdf_path,
        )
        assert isinstance(robot, FingeredSingleArmPyBulletRobot)
        robot.close_fingers()
        self.robot = robot

        # Create a holder (vention stand).
        self.robot_holder_id = create_pybullet_block(
            self.scene_spec.robot_holder_rgba,
            half_extents=self.scene_spec.robot_holder_half_extents,
            physics_client_id=self.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self.robot_holder_id,
            self.scene_spec.robot_holder_pose.position,
            self.scene_spec.robot_holder_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Create wheelchair.
        self._wheelchair_id = p.loadURDF(
            str(self.scene_spec.wheelchair_urdf_path),
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self._wheelchair_id,
            self.scene_spec.wheelchair_pose.position,
            self.scene_spec.wheelchair_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Create user.
        self._user_head = p.loadURDF(
            str(self.scene_spec.user_head_urdf_path),
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self._user_head,
            self.scene_spec.user_head_pose.position,
            self.scene_spec.user_head_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Create table.
        self.table_id = create_pybullet_cylinder(
            (1.0, 1.0, 1.0, 1.0),
            radius=self.scene_spec.table_radius,
            length=0.001,
            physics_client_id=self.physics_client_id
        )

        p.resetBasePositionAndOrientation(
            self.table_id,
            self.scene_spec.table_pose.position,
            self.scene_spec.table_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Create plate.
        self.plate_id = p.loadURDF(
            str(self.scene_spec.plate_urdf_path),
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )

        p.resetBasePositionAndOrientation(
            self.plate_id,
            BANISH_POSE.position,
            BANISH_POSE.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Create feeding utensil.
        self.utensil_id = p.loadURDF(
            str(self.scene_spec.utensil_urdf_path),
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self.utensil_id,
            self.scene_spec.utensil_pose.position,
            self.scene_spec.utensil_pose.orientation,
            physicsClientId=self.physics_client_id,
        )
        self.utensil_joints = []
        for i in range(p.getNumJoints(self.utensil_id)):
            joint_info = p.getJointInfo(self.utensil_id, i)
            if joint_info[2] != 4:  # Skip fixed joints.
                self.utensil_joints.append(i)

        # Create drink.
        self.drink_id = p.loadURDF(
            str(self.scene_spec.drink_urdf_path),
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self.drink_id,
            BANISH_POSE.position,
            BANISH_POSE.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Initialize held object.
        self.held_object_name: str | None = None
        self.held_object_tf: Pose | None = None

        # Initialize stage.
        self.current_stage = "acquisition"

        # Initialize user request.
        self.current_user_request: str | None = "food"
        self._total_user_requests = 0

        # Initialize user feedback.
        self.current_user_feedback: str | None = None

        # Initialize the occlusion scale.
        self._occlusion_scale = 0.0
        if self._hidden_spec and self._hidden_spec.occlusion_preference_scale > 0:
            self.set_occlusion_scale(self._hidden_spec.occlusion_preference_scale)
        self._occlusion_rays: set[int] | None = None

        # See get_joint_positions_from_known_ee_pose().
        self._known_ee_poses: dict[tuple[float, ...], JointPositions] = {}

        # Uncomment to debug.
        # if use_gui:
        #     while True:
        #         p.getMouseEvents(self.physics_client_id)

    def visualize_sample(self, pose: Pose, color: tuple[float, float, float, float]) -> None:
        """ Add a sphere to visualize a sample. """
        radius = 0.01
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color,
            physicsClientId=self.physics_client_id,
        )

        body_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=pose.position,
            baseOrientation=pose.orientation,
            physicsClientId=self.physics_client_id,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[FeedingState, dict[str, Any]]:

        # Reset the robot.
        self.robot.set_joints(self.scene_spec.initial_joints)
        self.held_object_name = None
        self.held_object_tf = None

        # Reset the stage.
        self.current_stage = "acquisition"

        # Reset the user feedback.
        self.current_user_feedback = None

        # Reset the tools.
        set_pose(self.utensil_id, self.scene_spec.utensil_pose, self.physics_client_id)

        # Randomly reset the plate.
        plate_x, plate_y = self._rng.uniform(
            low=self.scene_spec.plate_position_lower,
            high=self.scene_spec.plate_position_upper,
        )
        plate_z = self.scene_spec.plate_default_pose.position[2]
        plate_orn = self.scene_spec.plate_default_pose.orientation
        plate_pose = Pose((plate_x, plate_y, plate_z), plate_orn)
        set_pose(self.plate_id, plate_pose, self.physics_client_id)

        # Reset the user request. Alternate between food and drink.
        self._update_user_request()

        return self.get_state(), self._get_info()

    def get_state(self) -> FeedingState:
        """Get the current state of the environment."""
        # Get the joint positions of the robot.
        robot_joints = self.robot.get_joint_positions()

        # Get the plate pose.
        plate_pose = get_pose(self.plate_id, self.physics_client_id)

        # Get the drink pose.
        drink_pose = get_pose(self.drink_id, self.physics_client_id)

        # Create and return the FeedingState.
        assert self.current_user_request is not None
        state = FeedingState(
            robot_joints=robot_joints,
            plate_pose=plate_pose,
            drink_pose=drink_pose,
            held_object_name=self.held_object_name,
            held_object_tf=self.held_object_tf,
            stage=self.current_stage,
            user_request=self.current_user_request,
            user_feedback=self.current_user_feedback,
        )
        return state

    def set_state(self, state: FeedingState) -> None:
        """Set the current state of the environment."""
        # Update the robot and any held object.
        held_object_id = (
            self.get_object_id_from_name(self.held_object_name)
            if self.held_object_name
            else None
        )
        set_robot_joints_with_held_object(
            self.robot,
            self.physics_client_id,
            held_object_id,
            state.held_object_tf,
            state.robot_joints,
        )
        # Update the plate pose.
        set_pose(self.plate_id, state.plate_pose, self.physics_client_id)
        # Update the drink pose.
        set_pose(self.drink_id, state.drink_pose, self.physics_client_id)
        # Update the feedback, stage, request.
        self.current_stage = state.stage
        self.current_user_request = state.user_request
        self.current_user_feedback = state.user_feedback

    def _get_info(self) -> dict[str, Any]:
        """Get additional information about the environment."""
        return {}

    def step(
        self, action: FeedingAction
    ) -> tuple[FeedingState, float, bool, bool, dict[str, Any]]:

        held_object_id = (
            self.get_object_id_from_name(self.held_object_name)
            if self.held_object_name
            else None
        )

        done = False

        if isinstance(action, MoveToJointPositions):
            current_joints = self.robot.get_joint_positions()
            new_joints = list(current_joints)
            new_joints[:7] = action.joint_positions
            set_robot_joints_with_held_object(
                self.robot,
                self.physics_client_id,
                held_object_id,
                self.held_object_tf,
                new_joints,
            )
            if self._use_gui:
                self._pause_gui(0.25)
        elif isinstance(action, CloseGripper):
            self.robot.close_fingers()
        elif isinstance(action, MoveToEEPose):
            self._move_to_ee_pose(action.pose)
            ee_pose_tuple = self._pose_to_hashable_tuple(action.pose)
            self._known_ee_poses[ee_pose_tuple] = self.robot.get_joint_positions()
        elif isinstance(action, GraspTool):
            self._execute_grasp_tool(action.tool)
        elif isinstance(action, UngraspTool):
            self._execute_ungrasp_tool()
        elif isinstance(action, MovePlate):
            if self._use_gui:
                for plate_pose in iter_between_poses(
                    get_pose(self.plate_id, self.physics_client_id),
                    action.plate_pose,
                    include_start=False,
                ):
                    set_pose(self.plate_id, plate_pose, self.physics_client_id)
                    self._pause_gui(0.1)
            else:
                set_pose(self.plate_id, action.plate_pose, self.physics_client_id)
        elif isinstance(action, MoveDrink):
            if self._use_gui:
                for drink_pose in iter_between_poses(
                    get_pose(self.drink_id, self.physics_client_id),
                    action.drink_pose,
                    include_start=False,
                ):
                    set_pose(self.drink_id, drink_pose, self.physics_client_id)
                    self._pause_gui(0.1)
            else:
                set_pose(self.drink_id, action.drink_pose, self.physics_client_id)
        elif isinstance(action, WaitForUserInput):
            if action.user_input == "done":
                done = True
                self.current_stage = "acquisition"
            elif action.user_input == "ready for transfer?":
                self.current_stage = "transfer"
        else:
            raise NotImplementedError("TODO")

        # Handle user feedback: if the current stage is transfer and there is
        # occlusion, tell the robot.
        self.current_user_feedback = None
        if (
            self.current_stage == "acquisition"
            and self._hidden_spec
            and self.robot_in_occlusion()
        ):
            self.current_user_feedback = "You're blocking my view!"
            logging.info("User feedback: %s", self.current_user_feedback)

        # Update the user request if done.
        if done:
            self._update_user_request()

        # Return the next state and default gym API stuff.
        return self.get_state(), 0.0, done, False, self._get_info()

    def get_object_id_from_name(self, name: str) -> int:
        """Get the PyBullet ID from the object name."""
        if name == "utensil":
            return self.utensil_id
        if name == "drink":
            return self.drink_id
        raise NotImplementedError(f"Object name '{name}' not recognized.")

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        camera_kwargs = self.scene_spec.get_camera_kwargs()
        img = capture_image(
            self.physics_client_id,
            **camera_kwargs,
            image_width=self.scene_spec.image_width,
            image_height=self.scene_spec.image_height,
        )
        # In non-render mode, PyBullet does not render background correctly.
        # We want the background to be black instead of white. Here, make the
        # assumption that all perfectly white pixels belong to the background
        # and manually swap in black.
        background_mask = (img == [255, 255, 255]).all(axis=2)
        img[background_mask] = 0

        return img  # type: ignore

    def set_occlusion_scale(self, scale: float) -> None:
        """Update the scale of the occlusion model."""
        assert 0 <= scale <= 1.0, "Occlusion scale must be in [0, 1]"
        self._occlusion_scale = scale

    def get_joint_positions_from_known_ee_pose(self, ee_pose: Pose) -> JointPositions:
        """Given an end effector pose that was previously commanded by the
        robot for MoveToEEPose, return the joint positions that resulted."""
        ee_pose_tuple = self._pose_to_hashable_tuple(ee_pose)
        assert ee_pose_tuple in self._known_ee_poses, f"Unknown ee_pose: {ee_pose}"
        return self._known_ee_poses[ee_pose_tuple]

    def _pose_to_hashable_tuple(self, pose: Pose) -> tuple[float, ...]:
        position_tuple = tuple(np.round(pose.position, decimals=5).tolist())
        orientation_tuple = tuple(np.round(pose.orientation, decimals=5).tolist())
        return position_tuple + orientation_tuple

    def _move_to_ee_pose(self, pose: Pose, max_control_time: float = 30.0) -> None:
        initial_fingers_positions = self.robot.get_joint_positions()[7:]

        joint_trajectory: list[JointPositions] = []

        start_time = time.time()
        target_reached = False
        held_object_id = (
            self.get_object_id_from_name(self.held_object_name)
            if self.held_object_name
            else None
        )
        while time.time() - start_time < max_control_time:
            current_pose = self.robot.get_end_effector_pose()
            if pose.allclose(current_pose, atol=1e-2):
                target_reached = True
                break
            current_joint_positions = self.robot.get_joint_positions()
            joint_trajectory.append(current_joint_positions)
            current_jacobian = self.robot.get_jacobian()
            target_positions = cartesian_control_step(
                current_joint_positions, current_jacobian, current_pose, pose
            )
            target_positions = np.concatenate(
                (target_positions, initial_fingers_positions)
            ).tolist()
            set_robot_joints_with_held_object(
                self.robot,
                self.physics_client_id,
                held_object_id,
                self.held_object_tf,
                target_positions,
            )

        if not target_reached:
            raise RuntimeError(
                "Sim cartesian controller: Failed to reach target pose in time"
            )

    def _execute_grasp_tool(self, tool: str) -> None:
        self.robot.set_finger_state(self.scene_spec.tool_grasp_fingers_value)
        self.held_object_name = tool
        finger_frame_id = self.robot.link_from_name("finger_tip")
        end_effector_link_id = self.robot.link_from_name(self.robot.tool_link_name)
        finger_from_end_effector = get_relative_link_pose(
            self.robot.robot_id,
            finger_frame_id,
            end_effector_link_id,
            self.physics_client_id,
        )
        self.held_object_tf = finger_from_end_effector
        if tool == "utensil":
            assert self.held_object_tf.allclose(self.scene_spec.utensil_held_object_tf)
        if tool == "drink":
            assert self.held_object_tf.allclose(self.scene_spec.drink_held_object_tf)

    def _execute_ungrasp_tool(self) -> None:
        self.robot.close_fingers()
        self.held_object_name = None
        self.held_object_tf = None

    def _reset_drink_pose(self) -> None:
        for _ in range(1000):
            drink_x, drink_y = self._rng.uniform(
                low=self.scene_spec.drink_position_lower,
                high=self.scene_spec.drink_position_upper,
            )
            drink_z = self.scene_spec.drink_default_pose.position[2]
            drink_orn = self.scene_spec.drink_default_pose.orientation
            drink_pose = Pose((drink_x, drink_y, drink_z), drink_orn)
            set_pose(self.drink_id, drink_pose, self.physics_client_id)
            if not check_body_collisions(
                self.plate_id, self.drink_id, self.physics_client_id
            ):
                break
        else:
            raise RuntimeError("Failed to reset drink.")

    def _update_user_request(self) -> None:
        if self._total_user_requests % 4 < 2:
            self.current_user_request = "food"
            set_pose(self.drink_id, BANISH_POSE, self.physics_client_id)
        elif self._total_user_requests % 4 == 2:
            self.current_user_request = "prepare"
            self._reset_drink_pose()
        else:
            self.current_user_request = "drink"
        self._total_user_requests += 1

    def robot_in_occlusion(self) -> bool:
        """Check if the robot is in occlusion."""
        score = self.get_occlusion_score()
        return score >= 1.0 - self._occlusion_scale

    def get_occlusion_score(self) -> float:
        """A score between 0 and 1 where higher is more occluded."""

        # Check for occlusion following https://arxiv.org/pdf/2111.11401 (Eq 11).

        # The rays start in an array relative to the head position and point in
        # straight lines for a maximum distance.
        eye_pose = self.scene_spec.user_eyes_pose

        ray_from_positions = []
        ray_to_positions = []
        grid_size = self.scene_spec.occlusion_grid_size
        max_ray_length = self.scene_spec.occlusion_max_ray_length
        for r in range(grid_size):
            row_val = (r - grid_size // 2) * self.scene_spec.occlusion_grid_delta_r
            for c in range(grid_size):
                col_val = (c - grid_size // 2) * self.scene_spec.occlusion_grid_delta_c
                # Transform to world pose frame.
                ray_from = Pose((row_val, col_val, 0.0))
                ray_to = Pose((row_val, col_val, max_ray_length))
                eye_ray_from = multiply_poses(eye_pose, ray_from)
                eye_ray_to = multiply_poses(eye_pose, ray_to)
                ray_from_positions.append(eye_ray_from.position)
                ray_to_positions.append(eye_ray_to.position)

        ray_outputs = p.rayTestBatch(
            rayFromPositions=ray_from_positions,
            rayToPositions=ray_to_positions,
            physicsClientId=self.physics_client_id,
        )

        # Debug visualize the rays.
        if self._use_gui and self._occlusion_rays is None:
            self._occlusion_rays = set()
            for r in range(len(ray_from_positions)):
                ray_id = p.addUserDebugLine(
                    ray_from_positions[r],
                    ray_to_positions[r],
                    lineColorRGB=[1, 0, 0],
                    lineWidth=2,
                    physicsClientId=self.physics_client_id,
                )
                self._occlusion_rays.add(ray_id)

        # See equation 11 in paper.
        # NOTE: unlike the paper, we are primarily concerned with occlusion
        # during acquisition, so we actually give higher scores when the robot
        # is more in the line of SIGHT, as opposed to the paper, which considers
        # transfer, and gives lower scores for being in the line of the eye.
        alpha = self.scene_spec.occlusion_alpha
        sigma = self.scene_spec.occlusion_sigma
        score = 0.0
        for output in ray_outputs:
            if output[0] != -1:
                world_hit_pose = Pose(output[3])
                # Transform the hit position back into the eye frame.
                hit_pose = multiply_poses(eye_pose.invert(), world_hit_pose)
                # See equation 11 in paper.
                vec = np.array(hit_pose.position[:2])
                if np.isclose(hit_pose.position[2], 0.0):
                    point_score = 0.0
                else:
                    # See note above: in the paper this is 1 - [quantity].
                    point_score = np.exp(
                        -alpha
                        * np.transpose(vec)
                        @ sigma
                        @ vec
                        / (hit_pose.position[2] ** 2)
                    )
                score += point_score

        if score > 0:
            score /= len(ray_outputs)

        print("score:", score)

        return 0.499 if score > 0 else 0.0

    def _pause_gui(self, duration: float) -> None:
        if not self._use_gui:
            time.sleep(duration)
        else:
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < duration:
                p.getMouseEvents(self.physics_client_id)
