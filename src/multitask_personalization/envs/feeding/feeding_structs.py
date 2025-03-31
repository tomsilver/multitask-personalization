"""Data structures for the feeding environment."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions


@dataclass(frozen=True)
class FeedingState:
    """A state in the feeding environment."""

    robot_joints: JointPositions


FeedingAction: TypeAlias = JointPositions
