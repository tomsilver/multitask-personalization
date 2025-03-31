"""Data structures for the feeding environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from pybullet_helpers.joint import JointPositions


@dataclass(frozen=True)
class FeedingState:
    """A state in the feeding environment."""

    robot_joints: JointPositions


FeedingAction: TypeAlias = JointPositions
