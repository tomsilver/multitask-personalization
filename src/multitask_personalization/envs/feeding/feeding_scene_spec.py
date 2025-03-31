"""Scene specification for feeding environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions

from multitask_personalization.structs import PublicSceneSpec


@dataclass(frozen=True)
class FeedingSceneSpec(PublicSceneSpec):
    """Scene specification for the assistive feeding environment."""

    floor_position: tuple[float, float, float] = (0, 0, -0.4)
    floor_urdf: Path = Path(__file__).parent / "assets" / "floor" / "floor.urdf"

    # Robot.
    robot_name: str = "kinova-gen3"
    robot_urdf_path: Path = Path(__file__).parent / "assets" / "robot" / "robot.urdf"
    robot_base_pose: Pose = Pose(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    initial_joints: JointPositions = field(default_factory=lambda :
                                           [0.0, -0.34903602299465675, -3.141591055693139, -2.5482592711638783, 0.0, -0.872688061814757, 1.57075917569769, 0.8, 0.8, 0.8, 0.8, -0.8, -0.8])
    tool_frame_to_finger_tip: Pose = Pose(
        (0.0, 0.0, 0.05955),
        (0.0, 0.0, 0.0, 1.0),
    )
    # end_effector_link to camera_color_optical_frame
    camera_pose: Pose = Pose(
        (-0.046, 0.083, 0.125),
        (0.006, 0.708, 0.005, 0.706),
    )

    def get_camera_kwargs(self) -> dict[str, Any]:
        """Get camera kwargs."""
        return {
            "camera_target": (0.0, 0.0, 0.2),
            "camera_distance": 2.5,
            "camera_pitch": -35,
            "camera_yaw": -35,
        }
