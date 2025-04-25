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
    robot_holder_pose: Pose = Pose((0.0, 0.0, -0.34))
    robot_holder_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    robot_holder_half_extents: tuple[float, float, float] = (0.10, 0.10, 0.33)

    # Wheelchair.
    wheelchair_pose: Pose = Pose((-0.2, 0.45, -0.06), (0.0, 0.0, 0.0, 1.0))
    wheelchair_relative_head_pose: Pose = Pose((0.0, -0.25, 0.75), (0.0, 0.0, 0.0, 1.0))
    wheelchair_urdf_path: Path = (
        Path(__file__).parent / "assets" / "wheelchair" / "wheelchair.urdf"
    )
    wheelchair_mesh_path: Path = (
        Path(__file__).parent / "assets" / "wheelchair" / "wheelchair.obj"
    )

    # User head.
    user_head_pose: Pose = Pose((-0.4, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))

    # User eyes.
    user_eyes_relative_pose: Pose = Pose((0.0, 0.1, 0.1))

    user_head_urdf_path: Path = (
        Path(__file__).parent / "assets" / "head_models" / "mouth_open.urdf"
    )

    # Table.
    table_pose: Pose = Pose((0.45, 0.5, 0.15))
    table_half_extents: tuple[float, float, float] = (0.35, 0.4, 0.001)
    # table_radius: float = 1.2/2

    # Plate.
    plate_default_pose: Pose = Pose((0.4, 0.3, 0.17))
    plate_urdf_path: Path = Path(__file__).parent / "assets" / "plate" / "plate_with_holder.urdf"
    plate_mesh_path: Path = Path(__file__).parent / "assets" / "plate" / "plate_with_holder.obj"
    plate_radius: float = 0.15

    # Utensil.
    utensil_urdf_path: Path = (
        Path(__file__).parent / "assets" / "feeding_utensil" / "feeding_utensil.urdf"
    )
    utensil_inside_mount: Pose = Pose((0.242, -0.077, 0.07), (-1, 0, 0, 0))
    utensil_above_mount: Pose = Pose((0.242, -0.077, 0.17), (-1, 0, 0, 0))
    utensil_outside_above_mount_pos: JointPositions = field(
        default_factory=lambda: [
            -0.2692035082617874,
            0.4127082432063301,
            -2.513398492494741,
            -1.9930522355357558,
            -0.31928105676741936,
            -0.8392446174777604,
            0.5472652562309106,
        ]
    )

    # Drink.
    drink_urdf_path: Path = (
        Path(__file__).parent / "assets" / "drinking_utensil" / "drinking_utensil.urdf"
    )
    drink_default_pose: Pose = Pose(
        (0.55, 0.6, 0.35), (0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0)
    )
    drink_gaze_pos: JointPositions = field(
        default_factory=lambda: [
            -0.004187021865822871,
            0.6034579885210962,
            -3.1259047705564633,
            -2.3538005746884725,
            0.01149092320739253,
            1.3411586039000891,
            1.6825233913747728,
        ]
    )
    drink_staging_pos: JointPositions = field(
        default_factory=lambda: [
            -2.5860902733967808,
            -1.105096803823792,
            -1.0315333702969696,
            -1.3979449215077393,
            -0.7852325147776451,
            -0.8370922506847585,
            -2.7182634909296315,
        ]
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

    # Skill waypoints.
    retract_pos: JointPositions = field(
        default_factory=lambda: [
            0.0,
            -0.34903602299465675,
            -3.141591055693139,
            -2.5482592711638783,
            0.0,
            -0.872688061814757,
            1.57075917569769,
        ]
    )
    utensil_above_mount_pos: JointPositions = field(
        default_factory=lambda: [
            -0.3081224117999879,
            0.1449308244187662,
            -2.4515079603418446,
            -2.3539334664268674,
            -0.14376009880356744,
            -0.6872590793313744,
            0.5028097739444904,
        ]
    )
    utensil_outside_mount: Pose = Pose((0.372, -0.077, 0.07), (-1, 0, 0, 0))
    utensil_outside_above_mount: Pose = Pose((0.372, -0.077, 0.17), (-1, 0, 0, 0))
    before_transfer_pos: JointPositions = field(
        default_factory=lambda: [
            -2.86554642,
            -1.61951779,
            -2.60986085,
            -1.37302839,
            1.11779249,
            -1.18028264,
            2.05515862,
        ]
    )
    # NOTE: this value is with respect to the init plate pose. We will transform
    # it when the plate moves.
    above_plate_pos: JointPositions = field(
        default_factory=lambda: [
            -2.86495014,
            -1.61460533,
            -2.6115943,
            -1.37673391,
            1.11842806,
            -1.17904586,
            -2.6957422,
        ]
    )
    before_transfer_pose: Pose = Pose((0.504, 0.303, 0.529), (0.0, 0.707, 0.707, 0))
    outside_mouth_transfer_pose: Pose = Pose((0.0, 0.5, 0.67), (0.0, 0.707, 0.707, 0))

    drink_default_pre_grasp_pose: Pose = Pose(
        position=(0.56, 0.32, 0.26), orientation=(0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0)
    )
    drink_default_inside_bottom_pose: Pose = Pose(
        position=(0.56, 0.55, 0.26), orientation=(0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0)
    )
    drink_default_inside_top_pose: Pose = Pose(
        position=(0.56, 0.55, 0.34), orientation=(0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0)
    )
    drink_default_post_grasp_pose: Pose = Pose(
        position=(0.56, 0.55, 0.6), orientation=(0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0)
    )
    drink_before_transfer_pos: JointPositions = field(
        default_factory=lambda: [
            2.947,
            -1.294,
            -1.579,
            -1.316,
            1.263,
            -0.094,
            1.789,
        ]
    )
    drink_before_transfer_pose: Pose = Pose(
        (0.504, 0.58, 0.529), (0.0, 0.707, 0.707, 0)
    )

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