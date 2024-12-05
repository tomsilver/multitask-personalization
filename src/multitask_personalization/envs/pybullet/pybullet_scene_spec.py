"""Defines the specification of tasks in the pybullet environment."""

from __future__ import annotations
from pathlib import Path

from dataclasses import dataclass, field

import numpy as np
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

    floor_position: tuple[float, float, float] = (0, 0, -0.4)
    floor_urdf: Path = Path(__file__).parent / "assets" / "wood_floor.urdf"

    wall_poses: list[Pose] = field(
        default_factory=lambda: [
            Pose.from_rpy((0.0, 1.25, 0.0), (np.pi / 2, 0.0, np.pi / 2)),
            Pose.from_rpy((-1.25, 0.0, 0.0), (np.pi / 2, 0.0, 0.0)),
            Pose.from_rpy((4.25, 0.0, 0.0), (np.pi / 2, 0.0, np.pi)),
            Pose.from_rpy((0.0, 0.0, 3.0), (0.0, np.pi / 2, 0.0)),
    ])
    wall_half_extents: tuple[float, float, float] = (0.1, 3.0, 5.0)
    wall_texture: Path = Path(__file__).parent / "assets" / "tiled_wall_texture.jpg"

    robot_name: str = "kinova-gen3"  # must be 7-dof and have fingers
    robot_base_pose: Pose = Pose((0.0, 0.0, 0.0))
    initial_joints: JointPositions = field(
        default_factory=lambda: [
            1.0,
            -0.71,
            -np.pi,
            -2.3,
            0.0,
            0.0,
            -np.pi / 2,
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
    robot_stand_rgba: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 1.0)
    robot_stand_radius: float = 0.1
    robot_stand_length: float = 0.4

    human_spec: HumanSpec = HumanSpec()

    wheelchair_base_pose: Pose = Pose(position=(1.5, 0.5, -0.33))
    wheelchair_rgba: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)

    table_pose: Pose = Pose(position=(-0.75, 0.0, -0.2))
    table_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    table_half_extents: tuple[float, float, float] = (0.1, 0.3, 0.2)

    object_pose: Pose = Pose(position=(-1000, -1000, 0.05))
    object_rgba: tuple[float, float, float, float] = (0.9, 0.6, 0.3, 1.0)
    object_radius: float = 0.025
    object_length: float = 0.1

    camera_distance: float = 2.0

    shelf_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    shelf_width: float = 0.8
    shelf_height: float = 0.1
    shelf_depth: float = 0.3
    shelf_spacing: float = 0.3
    shelf_support_width: float = 0.05
    shelf_num_layers: int = 3
    shelf_support_height = (shelf_num_layers - 1) * shelf_spacing + (
        shelf_num_layers - 1
    ) * shelf_height
    shelf_pose: Pose = Pose(position=(0.0, 0.75, -shelf_support_height / 2))
    surface_texture: Path = Path(__file__).parent / "assets" / "dark_wood_texture.jpg"

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
                + shelf_support_width + 0.05,
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

    surface_dust_patch_size: int = 2  # dust arrays will be this number ^ 2
    surface_max_dust: float = 1.0
    max_dust_clean_threshold: float = 0.5
    dirty_patch_penalty: float = -0.5
    surface_dust_delta: float = 1e-2
    surface_dust_height: float = 1e-3
    dust_color: tuple[float, float, float] = (0.7, 0.5, 0.2)

    duster_head_forward_length: float = 0.04
    duster_head_long_length: float = 0.075
    duster_head_up_down_length: float = 0.04
    duster_head_rgba: tuple[float, float, float, float] = (155 / 255, 126 / 255, 189 / 255, 1.0)
    duster_pole_radius: float = 0.01
    duster_pole_height: float = 0.35
    duster_pole_rgba: tuple[float, float, float, float] = (59 / 255, 30 / 255, 84 / 255, 1.0)
    duster_pole_offset: tuple[float, float, float] = (
        duster_head_forward_length - duster_pole_radius,
        0,
        duster_head_up_down_length + duster_pole_height / 2,
    )
    duster_pose: Pose = Pose(
        position=(table_pose.position[0], 0.2, duster_head_up_down_length)
    )

    cleaning_admonishment_min_time_interval: int = 25

    @property
    def duster_grasp(self) -> Pose:
        """Hardcode a good relative grasp for the duster."""
        return Pose.from_rpy(
            (
                self.duster_pole_offset[0] + 2 * self.duster_pole_radius,
                0,
                self.duster_head_up_down_length + 0.8 * self.duster_pole_height,
            ),
            (np.pi / 2, np.pi, -np.pi / 2),
        )


@dataclass(frozen=True)
class HiddenSceneSpec:
    """Defines hidden parameters for a pybullet environment."""

    book_preferences: str  # a natural language description
    rom_model: ROMModel
    surfaces_robot_can_clean: list[tuple[str, int]]
