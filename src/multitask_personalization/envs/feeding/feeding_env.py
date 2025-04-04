"""Assistive feeding environment in pybullet."""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium.core import RenderFrame
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import Pose, get_pose, set_pose
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.inverse_kinematics import set_robot_joints_with_held_object
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.link import get_relative_link_pose
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block
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
    MovePlate,
    MoveToEEPose,
    MoveToJointPositions,
    UngraspTool,
    WaitForUserInput,
)
from multitask_personalization.envs.feeding.feeding_utils import cartesian_control_step


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
        self.table_id = p.loadURDF(
            str(self.scene_spec.table_urdf_path),
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
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
            self.scene_spec.plate_init_pose.position,
            self.scene_spec.plate_init_pose.orientation,
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

        # Initialize held object.
        self.held_object_name: str | None = None
        self.held_object_tf: Pose | None = None

        # Uncomment to debug.
        # if use_gui:
        #     while True:
        #         p.getMouseEvents(self.physics_client_id)

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

        # Reset the tools.
        set_pose(self.utensil_id, self.scene_spec.utensil_pose, self.physics_client_id)

        # Reset the plate.
        set_pose(self.plate_id, self.scene_spec.plate_init_pose, self.physics_client_id)

        return self.get_state(), self._get_info()

    def get_state(self) -> FeedingState:
        """Get the current state of the environment."""
        # Get the joint positions of the robot.
        robot_joints = self.robot.get_joint_positions()

        # Get the plate pose.
        plate_pose = get_pose(self.plate_id, self.physics_client_id)

        # Create and return the FeedingState.
        state = FeedingState(
            robot_joints=robot_joints,
            plate_pose=plate_pose,
            held_object_name=self.held_object_name,
            held_object_tf=self.held_object_tf,
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
                time.sleep(5.0)  # visualize the motion in GUI mode
        elif isinstance(action, CloseGripper):
            self.robot.close_fingers()
        elif isinstance(action, MoveToEEPose):
            self._move_to_ee_pose(action.pose)
        elif isinstance(action, GraspTool):
            self._execute_grasp_tool(action.tool)
        elif isinstance(action, UngraspTool):
            self._execute_ungrasp_tool()
        elif isinstance(action, MovePlate):
            set_pose(self.plate_id, action.plate_pose, self.physics_client_id)
        elif isinstance(action, WaitForUserInput):
            if action.user_input == "done":
                done = True
        else:
            raise NotImplementedError("TODO")

        # Return the next state and default gym API stuff.
        return self.get_state(), 0.0, done, False, self._get_info()

    def get_object_id_from_name(self, name: str) -> int:
        """Get the PyBullet ID from the object name."""
        if name == "utensil":
            return self.utensil_id
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

    def _execute_ungrasp_tool(self) -> None:
        self.robot.close_fingers()
        self.held_object_name = None
        self.held_object_tf = None
