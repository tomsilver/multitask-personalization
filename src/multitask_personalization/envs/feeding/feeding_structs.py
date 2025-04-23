"""Data structures for the feeding environment."""

from __future__ import annotations

import abc
from dataclasses import dataclass

from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions


class FeedingObservation(abc.ABC):
    """An observation in the feeding environment."""


@dataclass
class FeedingInitializationQueryObservation(FeedingObservation):
    """An observation of the initial feeding context."""
    
    context: str
    table_type: str
    food_items: list[str]
    dips: list[str]
    bite_ordering_options: list[str]


@dataclass
class FeedingInitializationDatasetObservation(FeedingObservation):
    """An observation of the user's ground-truth preferences about initialization,
    along with the contextual observations again for convenience."""

    context: str
    table_type: str
    food_items: list[str]
    dips: list[str]
    bite_ordering_options: list[str]
    feeding_side: str
    bite_ordering: str
    ready_signal: str
    be_verbal: bool


class FeedingAction(abc.ABC):
    """An action in the feeding environment."""


@dataclass
class FeedingInitializationAction(FeedingAction):
    """A prediction of user preferences to initialize a meal."""

    feeding_side: str
    bite_ordering: str
    ready_signal: str
    be_verbal: bool



@dataclass(frozen=True)
class MoveToJointPositions(FeedingAction):
    """Move to specific joint positions."""

    joint_positions: JointPositions


@dataclass(frozen=True)
class MoveToEEPose(FeedingAction):
    """Move to specific end effector pose."""

    pose: Pose


@dataclass(frozen=True)
class MoveToLastJointPositionswithEEPose(FeedingAction):
    """Move to the last known joints where the end effector had this pose."""

    pose: Pose


class CloseGripper(FeedingAction):
    """Close the gripper."""


@dataclass(frozen=True)
class GraspTool(FeedingAction):
    """Grasp a given tool."""

    tool: str


class UngraspTool(FeedingAction):
    """Ungrasp the currently held tool."""


@dataclass(frozen=True)
class WaitForUserInput(FeedingAction):
    """Wait for user input."""

    user_input: str


@dataclass(frozen=True)
class MovePlate(FeedingAction):
    """Move the plate to a specific pose."""

    plate_pose: Pose


@dataclass(frozen=True)
class MoveDrink(FeedingAction):
    """Move the drink to a specific pose."""

    drink_pose: Pose
