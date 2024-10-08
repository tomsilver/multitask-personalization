"""A handover environment implemented in PyBullet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, Pose3D
from pybullet_helpers.joint import JointPositions
from tomsutils.spaces import EnumSpace

from multitask_personalization.envs.intake_process import IntakeProcess
from multitask_personalization.envs.mdp import MDP
from multitask_personalization.envs.task import Task
from multitask_personalization.structs import (
    CategoricalDistribution,
    Image,
)


@dataclass(frozen=True)
class _HandoverState:
    """A state in the handover environment."""

    robot_base: Pose
    robot_joints: JointPositions
    human_base: Pose
    human_joints: JointPositions

    @classmethod
    def get_dimension(cls) -> int:
        """Get the dimensionality of a handover state."""
        return 7 + 7 + 7 + 7

    def to_vec(self) -> NDArray[np.float32]:
        """Convert the state into a vector."""
        return np.hstack(
            [
                self.robot_base.position,
                self.robot_base.orientation,
                self.robot_joints,
                self.human_base.position,
                self.human_base.orientation,
                self.human_joints,
            ]
        )

    @classmethod
    def from_vec(cls, vec: NDArray[np.float32]) -> _HandoverState:
        """Create a state from a vector."""
        (
            robot_base_position_vec,
            robot_base_orientation_vec,
            robot_joints_vec,
            human_base_position_vec,
            human_base_orientation_vec,
            human_joints_vec,
        ) = np.split(vec, [3, 7, 14, 17, 21])
        robot_base = Pose(
            tuple(robot_base_position_vec), tuple(robot_base_orientation_vec)
        )
        robot_joints = robot_joints_vec.tolist()
        human_base = Pose(
            tuple(human_base_position_vec), tuple(human_base_orientation_vec)
        )
        human_joints = human_joints_vec.tolist()
        return _HandoverState(robot_base, robot_joints, human_base, human_joints)


_HandoverAction: TypeAlias = JointPositions | None  # None = ready for handover


class PyBulletHandoverMDP(MDP[_HandoverState, _HandoverAction]):
    """A handover environment implemented in PyBullet."""

    def __init__(self) -> None:
        # TODO load environment.
        import ipdb

        ipdb.set_trace()

    @property
    def state_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            -np.inf, np.inf, shape=(_HandoverState.get_dimension(),), dtype=np.float32
        )

    @property
    def action_space(self) -> gym.spaces.Space:
        # TODO check whether this is what I want it to be
        return gym.spaces.OneOf(
            (
                gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                EnumSpace([None]),
            )
        )

    def state_is_terminal(self, state: _HandoverState) -> bool:
        # TODO terminate if human reaches object.
        import ipdb

        ipdb.set_trace()

    def get_reward(
        self, state: _HandoverState, action: _HandoverAction, next_state: _HandoverState
    ) -> float:
        if self.state_is_terminal(next_state):
            return 1.0
        return 0.0

    def get_initial_state_distribution(
        self,
    ) -> CategoricalDistribution:
        raise NotImplementedError("Initial state distribution too large")

    def sample_initial_state(self, rng: np.random.Generator) -> _HandoverState:
        # TODO randomize robot and human initial positions
        import ipdb

        ipdb.set_trace()

    def get_transition_distribution(
        self, state: _HandoverState, action: _HandoverAction
    ) -> CategoricalDistribution:
        raise NotImplementedError("Sample transitions, don't enumerate them")

    def sample_next_state(
        self, state: _HandoverState, action: _HandoverAction, rng: np.random.Generator
    ) -> _HandoverState:
        # TODO implement deterministic transition distribution
        import ipdb

        ipdb.set_trace()

    def render_state(self, state: _HandoverState) -> Image:
        # TODO reset state and take image
        import ipdb

        ipdb.set_trace()


@dataclass(frozen=True)
class _ROMReachableQuestion:
    """Ask the person to try to reach a certain position."""

    position: Pose3D  # in absolute coordinates

    def __lt__(self, other: Any) -> bool:
        return str(self) < str(other)


_HandoverIntakeObs: TypeAlias = bool  # whether or not reaching is successful
_HandoverIntakeAction: TypeAlias = _ROMReachableQuestion


class PyBulletHandoverIntakeProcess(
    IntakeProcess[_HandoverIntakeObs, _HandoverIntakeAction]
):
    """Intake process for the pybullet handover environment."""

    def __init__(self, horizon: int) -> None:
        self._horizon = horizon

        # TODO load environment.
        import ipdb

        ipdb.set_trace()

    @property
    def observation_space(self) -> EnumSpace[_HandoverIntakeObs]:
        return EnumSpace([True, False])

    @property
    def action_space(self) -> EnumSpace[_HandoverIntakeAction]:
        return gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)

    @property
    def horizon(self) -> int:
        return self._horizon

    def get_observation_distribution(
        self,
        action: _HandoverIntakeAction,
    ) -> CategoricalDistribution[_HandoverIntakeObs]:
        # TODO use ground truth ROM model to check
        import ipdb

        ipdb.set_trace()


@dataclass
class PyBulletHandoverTask(Task):
    """The full handover task."""

    _id: str
    _intake_horizon: int

    @property
    def id(self) -> str:
        return self._id

    @property
    def mdp(self) -> PyBulletHandoverMDP:
        return PyBulletHandoverMDP()

    @property
    def intake_process(self) -> PyBulletHandoverIntakeProcess:
        return PyBulletHandoverIntakeProcess(self._intake_horizon)
