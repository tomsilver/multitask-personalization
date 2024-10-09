"""A handover environment implemented in PyBullet."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import assistive_gym.envs
import gymnasium as gym
import numpy as np
import pybullet as p
from assistive_gym.envs.agents.furniture import Furniture
from assistive_gym.envs.agents.human import Human
from assistive_gym.envs.human_creation import HumanCreation
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, Pose3D
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block
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


@dataclass(frozen=True)
class PyBulletHandoverSceneDescription:
    """Container for default hyperparameters."""

    robot_name: str = "kinova-gen3"  # must be 7-dof and have fingers
    robot_base_pose: Pose = Pose((-1.0, -0.5, 0.5))
    initial_joints: JointPositions = field(
        default_factory=lambda: [
            -4.3,
            -1.6,
            -4.8,
            -1.8,
            -1.4,
            -1.1,
            1.6,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    robot_max_joint_delta: float = 0.5

    robot_stand_pose: Pose = Pose((-1.0, -0.5, 0.3))
    robot_stand_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    robot_stand_half_extents: tuple[float, float, float] = (0.2, 0.2, 0.225)


class PyBulletHandoverSimulator:
    """A shared simulator used for both MDP and intake."""

    def __init__(
        self,
        scene_description: PyBulletHandoverSceneDescription,
        use_gui: bool = False,
        seed: int = 0,
    ) -> None:

        self._rng = np.random.default_rng(seed)
        self._scene_description = scene_description

        # Create the PyBullet client.
        if use_gui:
            self._physics_client_id = create_gui_connection(camera_yaw=180)
        else:
            self._physics_client_id = p.connect(p.DIRECT)

        # Create robot.
        robot = create_pybullet_robot(
            self._scene_description.robot_name,
            self._physics_client_id,
            base_pose=self._scene_description.robot_base_pose,
            control_mode="reset",
            home_joint_positions=self._scene_description.initial_joints,
        )
        assert isinstance(robot, FingeredSingleArmPyBulletRobot)
        robot.close_fingers()
        self.robot = robot

        # Create robot stand.
        self._robot_stand_id = create_pybullet_block(
            self._scene_description.robot_stand_rgba,
            half_extents=self._scene_description.robot_stand_half_extents,
            physics_client_id=self._physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self._robot_stand_id,
            self._scene_description.robot_stand_pose.position,
            self._scene_description.robot_stand_pose.orientation,
            physicsClientId=self._physics_client_id,
        )

        # Create human.
        human_creation = HumanCreation(
            self._physics_client_id, np_random=self._rng, cloth=False
        )
        self.human = Human([], controllable=False)
        self.human.init(
            human_creation,
            static_human_base=True,
            impairment="none",
            gender="male",
            config=None,
            id=self._physics_client_id,
            np_random=self._rng,
        )
        joints_positions = [
            (self.human.j_right_elbow, -90),
            (self.human.j_left_elbow, -90),
            (self.human.j_right_hip_x, -90),
            (self.human.j_right_knee, 80),
            (self.human.j_left_hip_x, -90),
            (self.human.j_left_knee, 80),
        ]
        joints_positions += [
            (self.human.j_head_x, 0.0),
            (self.human.j_head_y, 0.0),
            (self.human.j_head_z, 0.0),
        ]
        self.human.setup_joints(
            joints_positions, use_static_joints=True, reactive_force=None
        )

        # Create wheelchair.
        furniture = Furniture()
        directory = Path(assistive_gym.envs.__file__).parent / "assets"
        assert directory.exists()
        furniture.init(
            "wheelchair",
            directory,
            self._physics_client_id,
            self._rng,
            wheelchair_mounted=False,
        )

        while True:
            p.stepSimulation(physicsClientId=self._physics_client_id)


class PyBulletHandoverMDP(MDP[_HandoverState, _HandoverAction]):
    """A handover environment implemented in PyBullet."""

    def __init__(
        self,
        sim: PyBulletHandoverSimulator,
    ) -> None:
        self._sim = sim

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

    def __init__(self, horizon: int, sim: PyBulletHandoverSimulator) -> None:
        self._horizon = horizon
        self._sim = sim

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


class PyBulletHandoverTask(Task):
    """The full handover task."""

    def __init__(
        self,
        intake_horizon: int,
        scene_description: PyBulletHandoverSceneDescription | None = None,
        use_gui: bool = False,
    ) -> None:

        self._intake_horizon = intake_horizon
        self._use_gui = use_gui

        # Finalize the scene description.
        if scene_description is None:
            scene_description = PyBulletHandoverSceneDescription()
        self._scene_description = scene_description

        # Generate a shared PyBullet simulator.
        self._sim = PyBulletHandoverSimulator(scene_description, use_gui)

    @property
    def id(self) -> str:
        return "handover"

    @property
    def mdp(self) -> PyBulletHandoverMDP:
        return PyBulletHandoverMDP(self._sim)

    @property
    def intake_process(self) -> PyBulletHandoverIntakeProcess:
        return PyBulletHandoverIntakeProcess(self._intake_horizon, self._sim)
