"""Scene specification for feeding environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

import numpy as np
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, Pose3D
from pybullet_helpers.joint import JointPositions

from multitask_personalization.structs import PublicSceneSpec

def create_feeding_scene_description_from_config(config_file_path: str) -> FeedingSceneSpec:
    """Create a SceneDescription instance from a YAML configuration file."""
    # Load the YAML file
    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)

    # Process the configuration dictionary
    processed_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            value_type = value.get("type")
            values = value.get("values")

            if not value_type or values is None:
                raise ValueError(f"Key '{key}' is missing 'type' or 'values': {value}")

            if value_type == "path":
                # Handle paths
                processed_config[key] = Path(__file__).parent / "assets" / Path(values)
            elif value_type == "bool":
                # Handle booleans
                processed_config[key] = bool(values)
            elif value_type == "pose":
                # Handle 3D poses
                if len(values) != 7:
                    processed_config[key] = None
                    print(f"Set '{key}' to None due to invalid pose values.")
                else:
                    position = tuple(values[:3])
                    orientation = tuple(values[3:])
                    processed_config[key] = Pose(position, orientation)
            elif value_type == "joint_positions":
                # Handle joint positions
                processed_config[key] = values
            else:
                raise ValueError(f"Unknown type '{value_type}' for key '{key}'")
        else:
            raise ValueError(f"Unexpected value type for key '{key}': {type(value)}")
    
    # Create an instance of SceneDescription using the processed config
    return FeedingSceneSpec(**processed_config)

@dataclass(frozen=True)
class FeedingSceneSpec(PublicSceneSpec):
    """Scene specification for the assistive feeding environment."""


    # Variables that change over different environments
    room_path: Path
    table_path: Path
    table_spawn_pose: Pose
    spawn_tv: bool
    spawn_social: bool
    social_base_pose: Pose
    robot_holder_pose: Pose
    wheelchair_pose: Pose
    user_head_pose: Pose
    user_eyes_relative_pose: Pose
    table_pose: Pose
    plate_default_pose: Pose
    utensil_inside_mount: Pose
    drink_default_pose: Pose 
    drink_staging_pos: JointPositions
    before_transfer_pos: JointPositions
    above_plate_pos: JointPositions
    before_transfer_pose: Pose

    # TV 
    tv_base_path: Path = (
        Path(__file__).parent / "assets" / "tv_world"
    )
    tv_objects: list[Path] = field(
        default_factory=lambda: [
            Path("body_1.obj"),
            Path("body_3.obj"),
            Path("body_4.obj"),
            Path("body_5.obj"),
            Path("body_6.obj"),
            Path("body_7_on.obj")
        ]
    )

    # social partner
    social_base_path: Path = (
        Path(__file__).parent / "assets" / "tv_world"
    )
    social_objects: list[Path] = field(
        default_factory=lambda: [
            Path("human_body.obj"),
            Path("chair_base.obj"),
            Path("chair_legs.obj"),
        ]
    )

    # Robot.
    robot_name: str = "kinova-gen3"
    robot_urdf_path: Path = Path(__file__).parent / "assets" / "robot" / "robot.urdf"
    robot_base_pose: Pose = Pose(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    
    initial_joints: JointPositions = field(
        default_factory=lambda: [
            0.0,
            -0.34903602299465675,
            -3.141591055693139,
            -2.5482592711638783,
            0.0,
            -0.872688061814757,
            1.57075917569769,
            0.8,
            0.8,
            0.8,
            0.8,
            -0.8,
            -0.8,
        ]
    )
    tool_frame_to_finger_tip: Pose = Pose(
        (0.0, 0.0, 0.05955),
        (0.0, 0.0, 0.0, 1.0),
    )
    tool_grasp_fingers_value: float = 0.44
    # end_effector_link to camera_color_optical_frame
    camera_pose: Pose = Pose(
        (-0.046, 0.083, 0.125),
        (0.006, 0.708, 0.005, 0.706),
    )

    # Robot holder (vention stand).
    robot_holder_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    robot_holder_half_extents: tuple[float, float, float] = (0.10, 0.10, 0.33)

    # Wheelchair.
    wheelchair_urdf_path: Path = (
        Path(__file__).parent / "assets" / "wheelchair" / "wheelchair.urdf"
    )
    wheelchair_mesh_path: Path = (
        Path(__file__).parent / "assets" / "wheelchair" / "wheelchair.obj"
    )

    # User head.
    user_head_urdf_path: Path = (
        Path(__file__).parent / "assets" / "head_models" / "mouth_open.urdf"
    )

    # Table.
    table_half_extents: tuple[float, float, float] = (0.35, 0.4, 0.001)
    # table_radius: float = 1.2/2

    # Plate.
    plate_urdf_path: Path = Path(__file__).parent / "assets" / "plate" / "plate_with_holder.urdf"
    plate_mesh_path: Path = Path(__file__).parent / "assets" / "plate" / "plate_with_holder.obj"
    plate_radius: float = 0.15

    # Utensil.
    utensil_urdf_path: Path = (
        Path(__file__).parent / "assets" / "feeding_utensil" / "feeding_utensil.urdf"
    )

    # Drink.
    drink_urdf_path: Path = (
        Path(__file__).parent / "assets" / "drinking_utensil" / "drinking_utensil.urdf"
    )
    drink_radius: float = 0.15

    # Occlusion model hyperparameters.
    occlusion_grid_size: int = 5
    occlusion_grid_delta_r: float = 0.03
    occlusion_grid_delta_c: float = 0.075
    occlusion_max_ray_length: float = 10.0
    occlusion_alpha: float = 1.0
    occlusion_sigma: NDArray = np.eye(2)
    occlusion_points_of_interest: dict[str, Pose3D] = field(
        default_factory=lambda: {
        "front": (10.0, 0.4, 0.4),
        "left": (10.0, 4.0, 0.4),
        "right": (10.0, -4.0, 0.4),
    })

    # This is redundant, but it's convenient for the CSP solver.
    utensil_held_object_tf: Pose = Pose(position=(0.0, 0.0, 0.05955))
    drink_held_object_tf: Pose = Pose(
        position=(0.0, 0.0, 0.05955), orientation=(0, 0, 0, 1)
    )

    # Rendering.
    image_height: int = 1024
    image_width: int = 2400

    @property
    def utensil_pose(self):
        """The initial utensil pose."""
        return self.utensil_inside_mount.multiply(self.tool_frame_to_finger_tip)

    @property
    def user_eyes_pose(self) -> Pose:
        """The user eyes pose in the world frame based on the user head
        pose."""
        return self.user_head_pose.multiply(self.user_eyes_relative_pose)

    def get_camera_kwargs(self, user_view: bool = False) -> dict[str, Any]:
        """Get camera kwargs."""
        if user_view:
            return {
                "camera_target": (0.5, 0.5, 0.5),
                "camera_distance": 0.9,
                "camera_pitch": -15,
                "camera_yaw": -90,
            }
        else:
            return {
                "camera_target": (0.0, 0.0, 0.2),
                "camera_distance": 2.0,
                "camera_pitch": -35,
                "camera_yaw": 90,
            }