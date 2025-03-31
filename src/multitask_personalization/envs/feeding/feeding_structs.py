"""Data structures for the feeding environment."""

from __future__ import annotations

import abc
from dataclasses import dataclass

from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions


@dataclass(frozen=True)
class FeedingState:
    """A state in the feeding environment."""

    robot_joints: JointPositions
    held_obj_name: str | None
    held_obj_tf: Pose | None


class FeedingAction(abc.ABC):
    """An action in the feeding environment."""


@dataclass(frozen=True)
class MoveToJointPositions(FeedingAction):
    """Move to specific joint positions."""

    joint_positions: JointPositions


@dataclass(frozen=True)
class MoveToEEPose(FeedingAction):
    """Move to specific end effector pose."""

    pose: Pose


class CloseGripper(FeedingAction):
    """Close the gripper."""


@dataclass(frozen=True)
class GraspTool(FeedingAction):
    """Grasp a given tool."""

    tool: str

