"""Assistive feeding environment in pybullet."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pybullet as p
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block
from tomsutils.spaces import FunctionalSpace

from multitask_personalization.envs.feeding.feeding_structs import (
    FeedingAction,
    FeedingState,
    MoveToJointPositions,
)
from multitask_personalization.envs.feeding.feeding_hidden_spec import (
    FeedingHiddenSceneSpec,
)

from multitask_personalization.envs.feeding.feeding_scene_spec import FeedingSceneSpec


class FeedingEnv(gym.Env[FeedingState, FeedingAction]):
    """An assistive feeding environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

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
        self.action_space = FunctionalSpace(contains_fn=lambda action: isinstance(action, FeedingAction))

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
            self.scene_spec.plate_pose.position,
            self.scene_spec.plate_pose.orientation,
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

        return self.get_state(), self._get_info()

    def get_state(self) -> FeedingState:
        """Get the current state of the environment."""
        # Get the joint positions of the robot.
        robot_joints = self.robot.get_joint_positions()

        # Create and return the FeedingState.
        state = FeedingState(
            robot_joints=robot_joints,
        )
        return state
    
    def set_state(self, state: FeedingState) -> None:
        """Set the current state of the environment."""
        # Set the robot joints to the specified state.
        self.robot.set_joints(state.robot_joints)

    def _get_info(self) -> dict[str, Any]:
        """Get additional information about the environment."""
        return {}

    def step(
        self, action: FeedingAction
    ) -> tuple[FeedingState, float, bool, bool, dict[str, Any]]:

        if isinstance(action, MoveToJointPositions):
            self.robot.set_joints(action.joint_positions)
        else:
            raise NotImplementedError("TODO")

        # Return the next state and default gym API stuff.
        return self.get_state(), 0.0, False, False, self._get_info()
