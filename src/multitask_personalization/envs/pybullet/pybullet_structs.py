"""States, actions, and other data structures for the pybullet environment."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions


@dataclass(frozen=True)
class PyBulletState:
    """A state in the pybullet environment."""

    robot_base: Pose
    robot_joints: JointPositions
    human_base: Pose
    human_joints: JointPositions
    cup_pose: Pose
    duster_pose: Pose
    book_poses: list[Pose]
    book_descriptions: list[str]
    grasp_transform: Pose | None
    surface_dust_patches: dict[tuple[str, int], NDArray]
    held_object: str | None = None
    human_text: str | None = None


class GripperAction(Enum):
    """Open or close the gripper."""

    OPEN = 1
    CLOSE = 2


PyBulletAction: TypeAlias = tuple[int, JointPositions | GripperAction | None]  # OneOf


class PyBulletMission(abc.ABC):
    """Missions for the robot in the PyBullet environment."""

    @abc.abstractmethod
    def get_id(self) -> str:
        """Get a unique identifier for the mission."""

    @abc.abstractmethod
    def get_mission_command(self) -> str:
        """Return a natural language description of the mission."""

    @abc.abstractmethod
    def check_initiable(self, state: PyBulletState) -> bool:
        """Check if the mission can be initiated from the given state.

        For example, we can only ask the robot to "put away the held
        object" if it is holding something.
        """

    @abc.abstractmethod
    def check_complete(self, state: PyBulletState, action: PyBulletAction) -> bool:
        """Check if the mission is complete."""

    @abc.abstractmethod
    def step(
        self, state: PyBulletState, action: PyBulletAction
    ) -> tuple[str | None, float]:
        """Return text and a user satisfaction value for the transition."""
