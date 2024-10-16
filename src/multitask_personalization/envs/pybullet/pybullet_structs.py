"""States, actions, and other data structures for the pybullet environment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from pybullet_helpers.geometry import Pose, Pose3D
from pybullet_helpers.joint import JointPositions


@dataclass(frozen=True)
class _PyBulletState:
    """A state in the pybullet environment."""

    robot_base: Pose
    robot_joints: JointPositions
    human_base: Pose
    human_joints: JointPositions
    object_pose: Pose
    book_pose: Pose
    grasp_transform: Pose | None
    held_object: str | None = None


class _GripperAction(Enum):
    """Open or close the gripper."""

    OPEN = 1
    CLOSE = 2


_PyBulletAction: TypeAlias = tuple[int, JointPositions | _GripperAction | None]  # OneOf
_PyBulletIntakeObs: TypeAlias = bool  # whether or not reaching is successful
_PyBulletIntakeAction: TypeAlias = Pose3D  # test handover position
