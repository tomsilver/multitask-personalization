"""Defines the specification of tasks in the pybullet environment."""

from __future__ import annotations

from dataclasses import dataclass, field

from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions

from multitask_personalization.envs.pybullet.pybullet_human_spec import HumanSpec
from multitask_personalization.rom.models import ROMModel
from multitask_personalization.structs import PublicSceneSpec


@dataclass(frozen=True)
class PyBulletSceneSpec(PublicSceneSpec):
    """Defines the specification of scenes in the pybullet environment."""

    scene_name: str = "default"

    world_lower_bounds: tuple[float, float, float] = (-0.5, -0.5, 0.0)
    world_upper_bounds: tuple[float, float, float] = (0.5, 0.5, 0.0)

    robot_name: str = "kinova-gen3"  # must be 7-dof and have fingers
    robot_base_pose: Pose = Pose((0.0, 0.0, 0.0))
    initial_joints: JointPositions = field(
        default_factory=lambda: [
            -4.3,
            -1.6,
            -4.8,
            -1.8,
            -1.4,
            -1.1,
            1.6,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    robot_max_joint_delta: float = 0.5

    robot_stand_pose: Pose = Pose((0.0, 0.0, -0.2))
    robot_stand_rgba: tuple[float, float, float, float] = (0.3, 0.1, 0.1, 1.0)
    robot_stand_radius: float = 0.1
    robot_stand_length: float = 0.4

    human_spec: HumanSpec = HumanSpec()

    wheelchair_base_pose: Pose = Pose(position=(1.0, 0.5, -0.46))

    table_pose: Pose = Pose(position=(-0.5, 0.0, -0.2))
    table_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    table_half_extents: tuple[float, float, float] = (0.1, 0.3, 0.2)

    object_pose: Pose = Pose(position=(-1000, -1000, 0.05))
    object_rgba: tuple[float, float, float, float] = (0.9, 0.6, 0.3, 1.0)
    object_radius: float = 0.025
    object_length: float = 0.1

    camera_distance: float = 2.0

    shelf_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    shelf_width: float = 1.0
    shelf_height: float = 0.1
    shelf_depth: float = 0.3
    shelf_spacing: float = 0.4
    shelf_support_width: float = 0.05
    shelf_num_layers: int = 4
    shelf_support_height = (shelf_num_layers - 1) * shelf_spacing + (
        shelf_num_layers - 1
    ) * shelf_height
    shelf_pose: Pose = Pose(position=(0.0, 0.75, -shelf_support_height / 2))

    book_rgbas: tuple[tuple[float, float, float, float], ...] = (
        (0.8, 0.2, 0.2, 1.0),
        (0.2, 0.8, 0.2, 1.0),
        (0.2, 0.2, 0.8, 1.0),
    )
    book_half_extents: tuple[tuple[float, float, float], ...] = (
        (0.02, 0.05, 0.08),
        (0.02, 0.05, 0.08),
        (0.02, 0.05, 0.08),
    )
    book_poses: tuple[Pose, ...] = (
        Pose(
            position=(
                shelf_pose.position[0],
                shelf_pose.position[1],
                shelf_pose.position[2]
                + (shelf_num_layers - 2) * shelf_spacing
                + (shelf_num_layers - 2) * shelf_height
                + book_half_extents[0][2]
                + shelf_support_width,
            )
        ),
        Pose(
            position=(
                shelf_pose.position[0] - 10 * book_half_extents[0][0],
                shelf_pose.position[1],
                shelf_pose.position[2]
                + (shelf_num_layers - 2) * shelf_spacing
                + (shelf_num_layers - 2) * shelf_height
                + book_half_extents[1][2]
                + shelf_support_width,
            )
        ),
        Pose(
            position=(
                shelf_pose.position[0] + 10 * book_half_extents[0][0],
                shelf_pose.position[1],
                shelf_pose.position[2]
                + (shelf_num_layers - 2) * shelf_spacing
                + (shelf_num_layers - 2) * shelf_height
                + book_half_extents[2][2]
                + shelf_support_width,
            )
        ),
    )

    # NOTE: the side table and tray are effectively disabled for now.
    side_table_pose: Pose = Pose(position=(-1000, -1000, -0.1))
    side_table_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    side_table_half_extents: tuple[float, float, float] = (0.025, 0.1, 0.4)

    tray_half_extents: tuple[float, float, float] = (0.4, 0.2, 0.025)
    tray_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)

    surface_dust_patch_size: int = 2  # dust arrays will be this number ^ 2
    surface_max_dust: float = 1.0
    max_dust_clean_threshold: float = 0.5
    dirty_patch_penalty: float = -0.5
    surface_dust_delta: float = 1e-2
    surface_dust_visual_height: float = 1e-3
    dust_color: tuple[float, float, float] = (0.7, 0.5, 0.2)

    @property
    def tray_pose(self) -> Pose:
        """Calculate the tray pose relative to the side table pose."""
        return Pose(
            position=(
                self.side_table_pose.position[0]
                + -(self.tray_half_extents[0] - self.side_table_half_extents[0]),
                self.side_table_pose.position[1],
                self.side_table_pose.position[2] + self.side_table_half_extents[2],
            )
        )


@dataclass(frozen=True)
class HiddenSceneSpec:
    """Defines hidden parameters for a pybullet environment."""

    book_preferences: str  # a natural language description
    rom_model: ROMModel
