"""Data structures for the feeding environment."""

from __future__ import annotations

import abc
from dataclasses import dataclass

from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions


class FeedingObservation(abc.ABC):
    """An observation in the feeding environment."""


@dataclass
class FeedingObservationWithContext(FeedingObservation):
    """An observation with context."""
    
    context: str
    table_type: str
    food_items: list[str]
    dips: list[str]
    bite_ordering_options: list[str]

    def get_context_str(self) -> str:
        """Get all the contextual information necessary to make LLM-based decisions."""
        return f"""
Dining Context: {self.context}
Table Type: {self.table_type}
Food Items on Plate: {self.food_items}
Food Dips on Plate: {self.dips}
"""


@dataclass
class FeedingInitializationQueryObservation(FeedingObservationWithContext):
    """An observation of the initial feeding context."""


@dataclass
class FeedingInitializationDatasetObservation(FeedingObservationWithContext):
    """An observation of the user's ground-truth preferences about initialization,
    along with the contextual observations again for convenience."""

    feeding_side: str
    bite_ordering: str
    ready_signal: str
    be_verbal: bool


@dataclass
class FeedingOcclusionQueryObservation(FeedingObservationWithContext):
    """An observation of the feeding context plus plate and drink poses."""

    plate_pose: Pose
    drink_pose: Pose


@dataclass
class FeedingOcclusionDatasetObservation(FeedingObservationWithContext):
    """An observation of the feeding context plus plate and drink poses,
    along with the user's ground-truth preferences about occlusion."""

    plate_pose: Pose
    drink_pose: Pose
    occlusion: dict[str, dict[str, bool]]


class FeedingAction(abc.ABC):
    """An action in the feeding environment."""


@dataclass
class FeedingInitializationAction(FeedingAction):
    """A prediction of user preferences to initialize a meal."""

    feeding_side: str
    bite_ordering: str
    ready_signal: str
    be_verbal: bool


@dataclass
class FeedingPlateDrinkAction(FeedingAction):
    """An action to handle plate and drink poses."""
    
    plate_delta_xy: tuple[float, float]
    drink_delta_xy: tuple[float, float]
    before_transfer_pose: Pose
    before_transfer_pos: JointPositions
    above_plate_pos: JointPositions
    drink_grasp_pos: JointPositions
    occlusion_poi_relevance: dict[str, bool]


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
