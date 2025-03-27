"""Defines the specification of tasks in the pybullet environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions

from multitask_personalization.envs.pybullet.pybullet_human import HumanSpec
from multitask_personalization.envs.pybullet.pybullet_structs import PyBulletState
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

    bed_pose: Pose = Pose.from_rpy((2.4, 0, -0.45), (np.pi / 2, 0.0, 0.0))
    bed_urdf: Path = Path(__file__).parent / "assets" / "bed" / "bed.urdf"

    wall_poses: list[Pose] = field(
        default_factory=lambda: [
            Pose.from_rpy((0.0, 1.25, 0.0), (np.pi / 2, 0.0, np.pi / 2)),
            Pose.from_rpy((-1.25, 0.0, 0.0), (np.pi / 2, 0.0, 0.0)),
            Pose.from_rpy((4.25, 0.0, 0.0), (np.pi / 2, 0.0, np.pi)),
            Pose.from_rpy((0.0, 0.0, 3.0), (0.0, np.pi / 2, 0.0)),
        ]
    )
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

    table_pose: Pose = Pose(position=(-0.75, 0.0, -0.2))
    table_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    table_half_extents: tuple[float, float, float] = (0.1, 0.3, 0.2)

    num_side_tables: int = 3
    default_side_table_half_extents: tuple[float, float, float] = (0.1, 0.1, 0.2)
    side_table_spacing: float = 0.05

    object_pose: Pose = Pose(position=(-1000, -1000, 0.05))
    object_rgba: tuple[float, float, float, float] = (0.9, 0.6, 0.3, 1.0)
    object_radius: float = 0.025
    object_length: float = 0.1

    image_height: int = 1024
    image_width: int = 1600

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

    use_standard_books: bool = False
    num_books: int = 3
    default_book_half_extents: tuple[float, float, float] = (0.02, 0.05, 0.08)

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
    duster_head_rgba: tuple[float, float, float, float] = (
        155 / 255,
        126 / 255,
        189 / 255,
        1.0,
    )
    duster_pole_radius: float = 0.01
    duster_pole_height: float = 0.35
    duster_pole_rgba: tuple[float, float, float, float] = (
        59 / 255,
        30 / 255,
        84 / 255,
        1.0,
    )
    duster_pole_offset: tuple[float, float, float] = (
        duster_head_forward_length - duster_pole_radius,
        0,
        duster_head_up_down_length + duster_pole_height / 2,
    )
    duster_pose: Pose = Pose(
        position=(table_pose.position[0], 0.2, duster_head_up_down_length)
    )

    cleaning_feedback_min_time_interval: int = 25
    cleaning_mission_eval_prob: float = 0.1

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

    @property
    def side_table_half_extents(self) -> tuple[tuple[float, float, float], ...]:
        """The half extents for all side tables."""
        return tuple([self.default_side_table_half_extents] * self.num_side_tables)

    @property
    def side_table_poses(self) -> tuple[Pose, ...]:
        """The side table poses."""
        x = self.table_pose.position[0]
        y = (
            self.table_pose.position[1]
            - self.table_half_extents[1]
            - self.side_table_spacing
        )
        z = self.table_pose.position[2]

        poses: list[Pose] = []
        for side_table_half_extent in self.side_table_half_extents:
            y -= side_table_half_extent[1]
            pose = Pose((x, y, z))
            poses.append(pose)
            y -= side_table_half_extent[1] + self.side_table_spacing

        return tuple(poses)

    @property
    def book_half_extents(self) -> tuple[tuple[float, float, float], ...]:
        """The half extents for all books."""
        return tuple([self.default_book_half_extents] * self.num_books)

    @property
    def book_poses(self) -> tuple[Pose, ...]:
        """The initial book poses."""

        all_possible_poses: list[Pose] = []

        # Books on the middle shelf.
        x = self.shelf_pose.position[0]
        dx = 10 * self.default_book_half_extents[0]
        y = self.shelf_pose.position[1]
        z = (
            self.shelf_pose.position[2]
            + (self.shelf_num_layers - 2) * self.shelf_spacing
            + (self.shelf_num_layers - 2) * self.shelf_height
            + self.default_book_half_extents[2]
            + self.shelf_support_width
        )

        all_possible_poses.append(Pose((x - dx, y, z)))
        all_possible_poses.append(Pose((x, y, z)))
        all_possible_poses.append(Pose((x + dx, y, z)))

        # Books on the top shelf.
        z = (
            self.shelf_pose.position[2]
            + (self.shelf_num_layers - 1) * self.shelf_spacing
            + (self.shelf_num_layers - 1) * self.shelf_height
            + self.default_book_half_extents[2]
            + self.shelf_support_width
        )

        all_possible_poses.append(Pose((x - dx, y, z)))
        all_possible_poses.append(Pose((x, y, z)))
        all_possible_poses.append(Pose((x + dx, y, z)))

        # Books on the side tables.
        for side_table_pose, side_table_half_extent in zip(
            self.side_table_poses, self.side_table_half_extents, strict=True
        ):
            x = side_table_pose.position[0]
            y = side_table_pose.position[1]
            z = (
                side_table_pose.position[2]
                + side_table_half_extent[2]
                + self.default_book_half_extents[2]
            )
            rpy = (0, 0, np.pi / 2)
            all_possible_poses.append(Pose.from_rpy((x, y, z), rpy))

        assert len(all_possible_poses) >= self.num_books
        return tuple(all_possible_poses[: self.num_books])
    
    def get_camera_kwargs(self, state: PyBulletState | None = None, timestep: int | None = None) -> dict[str, Any]:

        # Views the whole scene.
        default = {
            "camera_target": (0.0, 0.0, 0.2),
            "camera_distance": 2.5,
            "camera_pitch": -35,
            "camera_yaw": -35,
        }

        if state is None or timestep is None:
            return default

        # Look at the cover of into the wild.
        bookshelf_waypoint1 = {
            "camera_target": (0.4, 0.65, 0.5),
            "camera_distance": 0.15,
            "camera_pitch": 0,
            "camera_yaw": 90,
        }

        # Look at the book shelf from the right side.
        bookshelf_waypoint2 = {
            "camera_target": (0.0, 0.65, 0.5),
            "camera_distance": 1.0,
            "camera_pitch": -15,
            "camera_yaw": 65,
        }

        # Look at the human while the robot is handing over a book.
        human_waypoint = {
            "camera_target": (self.human_spec.base_pose.position[0], self.human_spec.base_pose.position[1], self.human_spec.base_pose.position[2] + 0.1), 
            "camera_distance": 2.5,
            "camera_pitch": -15,
            "camera_yaw": -45,
        }

        # Look at the robot from behind.
        robot_waypoint = {
            "camera_target": (state.robot_base.position[0], state.robot_base.position[1], state.robot_base.position[2] + 0.5),
            "camera_distance": 1.0,
            "camera_pitch": -35,
            "camera_yaw": state.robot_base.rpy[2],
        }

        def _interpolate_waypoints(wp1, wp2, t):
            return {
                "camera_target": tuple(
                    np.array(wp1["camera_target"]) * (1 - t)
                    + np.array(wp2["camera_target"]) * t
                ),
                "camera_distance": wp1["camera_distance"] * (1 - t)
                + wp2["camera_distance"] * t,
                "camera_pitch": wp1["camera_pitch"] * (1 - t)
                + wp2["camera_pitch"] * t,
                "camera_yaw": wp1["camera_yaw"] * (1 - t) + wp2["camera_yaw"] * t,
            }

        # Sequence of (waypoints, relative pause, relative transition) time.
        script = [
            (bookshelf_waypoint1, 0.5, 1.0),
            (robot_waypoint, 1.0, 1.0),
            (human_waypoint, 1.0, 1.0),
            (default, None, None),
        ]
        fps = 25
        timestep_secs = timestep / fps
        t = 0
        for idx, (waypoint, relative_pause, relative_transition) in enumerate(script):
            if relative_pause is None or relative_transition is None:
                return waypoint  # End of script

            # Pause phase
            if t + relative_pause > timestep_secs:
                return waypoint
            t += relative_pause  # Advance time after pause

            # Transition phase
            if t + relative_transition > timestep_secs:
                interp_t = (timestep_secs - t) / relative_transition
                next_waypoint = script[idx + 1][0]
                return _interpolate_waypoints(waypoint, next_waypoint, interp_t)
            t += relative_transition  # Advance time after transition

        raise NotImplementedError


@dataclass(frozen=True)
class HiddenSceneSpec:
    """Defines hidden parameters for a pybullet environment."""

    missions: str  # handover-only, clean-only, all
    book_preferences: str  # a natural language description
    rom_model: ROMModel
    surfaces_robot_can_clean: list[tuple[str, int]]
