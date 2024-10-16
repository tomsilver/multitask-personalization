"""States, actions, and other data structures for the pybullet environment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
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
    grasp_transform: Pose | None

    @classmethod
    def get_dimension(cls) -> int:
        """Get the dimensionality of a pybullet state."""
        return 7 + 7 + 7 + 7 + 7 + 8

    def to_vec(self) -> NDArray[np.float32]:
        """Convert the state into a vector."""
        if self.grasp_transform is None:
            grasp_transform_vec = np.zeros(8, dtype=np.float32)
        else:
            grasp_transform_vec = np.hstack(
                [[1], self.grasp_transform.position, self.grasp_transform.orientation]
            )
        return np.hstack(
            [
                self.robot_base.position,
                self.robot_base.orientation,
                self.robot_joints,
                self.human_base.position,
                self.human_base.orientation,
                self.human_joints,
                self.object_pose.position,
                self.object_pose.orientation,
                grasp_transform_vec,
            ]
        )

    @classmethod
    def from_vec(cls, vec: NDArray[np.float32]) -> _PyBulletState:
        """Create a state from a vector."""
        (  # pylint: disable=unbalanced-tuple-unpacking
            robot_base_position_vec,
            robot_base_orientation_vec,
            robot_joints_vec,
            human_base_position_vec,
            human_base_orientation_vec,
            human_joints_vec,
            object_position_vec,
            object_orientation_vec,
            grasp_transform_vec,
        ) = np.split(vec, [3, 7, 14, 17, 21, 25, 28, 32])
        robot_base = Pose(
            tuple(robot_base_position_vec), tuple(robot_base_orientation_vec)
        )
        robot_joints = robot_joints_vec.tolist()
        human_base = Pose(
            tuple(human_base_position_vec), tuple(human_base_orientation_vec)
        )
        human_joints = human_joints_vec.tolist()
        object_position = Pose(
            tuple(object_position_vec), tuple(object_orientation_vec)
        )
        if np.isclose(grasp_transform_vec[0], 0.0):
            grasp_transform: Pose | None = None
        else:
            assert np.isclose(grasp_transform_vec[0], 1.0)
            grasp_transform = Pose(
                tuple(grasp_transform_vec[1:4]), tuple(grasp_transform_vec[4:])
            )
        return _PyBulletState(
            robot_base,
            robot_joints,
            human_base,
            human_joints,
            object_position,
            grasp_transform,
        )


class _GripperAction(Enum):
    """Open or close the gripper."""

    OPEN = 1
    CLOSE = 2


_PyBulletAction: TypeAlias = tuple[int, JointPositions | _GripperAction | None]  # OneOf
_PyBulletIntakeObs: TypeAlias = bool  # whether or not reaching is successful
_PyBulletIntakeAction: TypeAlias = Pose3D  # test handover position
