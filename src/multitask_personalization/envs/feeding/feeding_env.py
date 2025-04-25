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
    FeedingObservation,
    FeedingInitializationQueryObservation,
    FeedingInitializationAction,
    FeedingOcclusionQueryObservation,
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


class FeedingEnv(gym.Env[FeedingObservation, FeedingAction]):
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
        self.table_id = create_pybullet_block(
            (0.0, 0.0, 0.0, 0.0),
            half_extents=self.scene_spec.table_half_extents,
            physics_client_id=self.physics_client_id
        )

        # self.table_id = create_pybullet_cylinder(
        #     (1.0, 1.0, 1.0, 1.0),
        #     radius=self.scene_spec.table_radius,
        #     length=0.001,
        #     physics_client_id=self.physics_client_id
        # )

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

        # Show the occlusion rays.
        if self._use_gui:
            for point_of_interest in self.scene_spec.occlusion_points_of_interest:
                ray_from_positions, ray_to_positions = self.get_occlusion_rays(point_of_interest)
                for r in range(len(ray_from_positions)):
                    p.addUserDebugLine(
                        ray_from_positions[r],
                        ray_to_positions[r],
                        lineColorRGB=[1, 0, 0],
                        lineWidth=2,
                        physicsClientId=self.physics_client_id,
                    )


        # See get_joint_positions_from_known_ee_pose().
        self._known_ee_poses: dict[tuple[float, ...], JointPositions] = {}

        # Uncomment to debug.
        # if use_gui:
        #     while True:
        #         p.getMouseEvents(self.physics_client_id)

        # spawn room
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=str(self.scene_spec.room_path),
            meshScale=[0.1, 0.1, 0.1],
            rgbaColor=[1, 1, 1, 1],  # Let texture show
            physicsClientId=self.physics_client_id,
        )

        p.createMultiBody(
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[1, 0.8, -0.66],  # X, Y, Z in meters
            physicsClientId=self.physics_client_id,
        )

        # spawn table
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=str(self.scene_spec.table_path),
            meshScale=[0.1, 0.1, 0.1],
            rgbaColor=[1, 1, 1, 1],  # Let texture show
            physicsClientId=self.physics_client_id,
        )

        p.createMultiBody(
            baseVisualShapeIndex=visual_shape_id,
            basePosition=self.scene_spec.table_spawn_pose.position,  # X, Y, Z in meters
            baseOrientation=self.scene_spec.table_spawn_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # spawn TV
        if self.scene_spec.spawn_tv:
            for body in self.scene_spec.tv_objects:
                visual_shape_id = p.createVisualShape(
                    shapeType=p.GEOM_MESH,
                    fileName=str(self.scene_spec.tv_base_path / body),
                    meshScale=[0.1, 0.1, 0.1],
                    rgbaColor=[1, 1, 1, 1],  # Let texture show
                    physicsClientId=self.physics_client_id,
                )

                p.createMultiBody(
                    baseVisualShapeIndex=visual_shape_id,
                    basePosition=[1, 0.8, -0.66],  # X, Y, Z in meters
                    physicsClientId=self.physics_client_id,
                )

        # spawn social 
        if self.scene_spec.spawn_social:
            for body in self.scene_spec.social_objects:
                visual_shape_id = p.createVisualShape(
                    shapeType=p.GEOM_MESH,
                    fileName=str(self.scene_spec.social_base_path / body),
                    meshScale=[0.1, 0.1, 0.1],
                    rgbaColor=[1, 1, 1, 1],  # Let texture show
                    physicsClientId=self.physics_client_id,
                )

                p.createMultiBody(
                    baseVisualShapeIndex=visual_shape_id,
                    basePosition=self.scene_spec.social_base_pose.position,  # X, Y, Z in meters
                    baseOrientation= self.scene_spec.social_base_pose.orientation,
                    physicsClientId=self.physics_client_id,
                )

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
    ) -> tuple[FeedingObservation, dict[str, Any]]:
        raise NotImplementedError("This environment is only being used as a simulator right now, not an actual environment.")

    def step(
        self, action: FeedingAction
    ) -> tuple[FeedingObservation, float, bool, bool, dict[str, Any]]:
        raise NotImplementedError("This environment is only being used as a simulator right now, not an actual environment.")

    def get_object_id_from_name(self, name: str) -> int:
        """Get the PyBullet ID from the object name."""
        if name == "utensil":
            return self.utensil_id
        if name == "drink":
            return self.drink_id
        if name == "plate":
            return self.plate_id
        raise NotImplementedError(f"Object name '{name}' not recognized.")

    def render(self, user_view: bool = False) -> RenderFrame | list[RenderFrame] | None:
        camera_kwargs = self.scene_spec.get_camera_kwargs(user_view=user_view)
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
    
    def get_occlusion_rays(self, point_of_interest: str):
        eye_pose = self.scene_spec.user_eyes_pose
        target_position = self.scene_spec.occlusion_points_of_interest[point_of_interest]

        ray_from_positions = []
        ray_to_positions = []
        grid_size = self.scene_spec.occlusion_grid_size
        max_ray_length = self.scene_spec.occlusion_max_ray_length

        # Vector from eye to target
        eye_pos = np.array(eye_pose.position)
        target_vec = np.array(target_position) - eye_pos
        target_dir = target_vec / np.linalg.norm(target_vec)

        for r in range(grid_size):
            row_val = (r - grid_size // 2) * self.scene_spec.occlusion_grid_delta_r
            for c in range(grid_size):
                col_val = (c - grid_size // 2) * self.scene_spec.occlusion_grid_delta_c
                # Offset ray origins in local eye frame
                local_offset = Pose((row_val, col_val, 0.0))
                ray_from = multiply_poses(eye_pose, local_offset).position
                ray_to = ray_from + max_ray_length * target_dir
                ray_from_positions.append(ray_from)
                ray_to_positions.append(ray_to)

        return ray_from_positions, ray_to_positions

    def get_occlusion_score(self, point_of_interest: str) -> float:
        """A score between 0 and 1 where higher is more occluded."""

        # Check for occlusion following https://arxiv.org/pdf/2111.11401 (Eq 11).
        ray_from_positions, ray_to_positions = self.get_occlusion_rays(point_of_interest)

        ray_outputs = p.rayTestBatch(
            rayFromPositions=ray_from_positions,
            rayToPositions=ray_to_positions,
            physicsClientId=self.physics_client_id,
        )

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
                hit_pose = multiply_poses(self.scene_spec.user_eyes_pose.invert(), world_hit_pose)
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

        print(f"score for POI={point_of_interest}:", score)

        # return 0.499 if score > 0 else 0.0
        return score

    def _pause_gui(self, duration: float) -> None:
        if not self._use_gui:
            time.sleep(duration)
        else:
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < duration:
                p.getMouseEvents(self.physics_client_id)
