"""Assistive feeding environment in pybullet."""

from __future__ import annotations

import logging
import pickle as pkl
from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium.core import RenderFrame
from numpy.typing import NDArray
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import (
    Pose,
    get_half_extents_from_aabb,
    get_pose,
    multiply_poses,
    rotate_pose,
    set_pose,
)
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.inverse_kinematics import (
    check_body_collisions,
    check_collisions_with_held_object,
)
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.link import get_link_pose
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.kinova import KinovaGen3RobotiqGripperPyBulletRobot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block, create_pybullet_cylinder
from tomsutils.spaces import EnumSpace

from multitask_personalization.envs.feeding.feeding_structs import (
    FeedingAction,
    FeedingState,
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

        # Uncomment to debug.
        if use_gui:
            while True:
                p.getMouseEvents(self.physics_client_id)

                