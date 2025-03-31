"""Scene specification for feeding environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions

from multitask_personalization.structs import PublicSceneSpec


@dataclass(frozen=True)
class FeedingSceneSpec(PublicSceneSpec):
    """Scene specification for the assistive feeding environment."""

    floor_position: tuple[float, float, float] = (0, 0, -0.66)
    floor_urdf: Path = Path(__file__).parent / "assets" / "floor" / "floor.urdf"

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
    wheelchair_pose: Pose = Pose((-0.3, 0.45, -0.06), (0.0, 0.0, 0.0, 1.0))
    wheelchair_relative_head_pose: Pose = Pose((0.0, -0.25, 0.75), (0.0, 0.0, 0.0, 1.0))
    wheelchair_urdf_path: Path = (
        Path(__file__).parent / "assets" / "wheelchair" / "wheelchair.urdf"
    )
    wheelchair_mesh_path: Path = (
        Path(__file__).parent / "assets" / "wheelchair" / "wheelchair.obj"
    )

    # User head.
    user_head_pose: Pose = Pose((-0.4, 0.5, 0.67), (0.5, 0.5, 0.5, 0.5))

    user_head_urdf_path: Path = (
        Path(__file__).parent / "assets" / "head_models" / "mouth_open.urdf"
    )

    # Table.
    table_pose: Pose = Pose((0.35, 0.45, 0.15))
    table_urdf_path: Path = Path(__file__).parent / "assets" / "table" / "table.urdf"
    table_mesh_path: Path = Path(__file__).parent / "assets" / "table" / "table.obj"

    # Plate.
    plate_pose: Pose = Pose((0.3, 0.25, 0.16))
    plate_urdf_path: Path = Path(__file__).parent / "assets" / "plate" / "plate.urdf"
    plate_mesh_path: Path = Path(__file__).parent / "assets" / "plate" / "plate.obj"

    # Utensil.
    utensil_urdf_path: Path = (
        Path(__file__).parent / "assets" / "feeding_utensil" / "feeding_utensil.urdf"
    )
    utensil_inside_mount: Pose = Pose((0.242, -0.077, 0.07), (-1, 0, 0, 0))

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

    @property
    def utensil_pose(self):
        return self.utensil_inside_mount.multiply(self.tool_frame_to_finger_tip)

    def get_camera_kwargs(self) -> dict[str, Any]:
        """Get camera kwargs."""
        return {
            "camera_target": (0.0, 0.0, 0.2),
            "camera_distance": 2.5,
            "camera_pitch": -35,
            "camera_yaw": -35,
        }
