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
    floor_urdf: Path = Path(__file__).parent.parent / "pybullet" / "assets" / "wood_floor.urdf"

    def get_camera_kwargs(self) -> dict[str, Any]:
        """Get camera kwargs."""
        return {
            "camera_target": (0.0, 0.0, 0.2),
            "camera_distance": 2.5,
            "camera_pitch": -35,
            "camera_yaw": -35,
        }
