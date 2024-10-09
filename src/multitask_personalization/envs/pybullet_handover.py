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
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import Pose, Pose3D
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.link import get_link_pose
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
        (  # pylint: disable=unbalanced-tuple-unpacking
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


_HandoverAction: TypeAlias = tuple[int, JointPositions | None]  # see OneOf


@dataclass(frozen=True)
class PyBulletHandoverSceneDescription:
    """Container for default hyperparameters."""

    robot_name: str = "kinova-gen3"  # must be 7-dof and have fingers
    robot_base_pose: Pose = Pose((0.0, 0.0, 0.0))
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

    robot_stand_pose: Pose = Pose((0.0, 0.0, -0.2))
    robot_stand_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    robot_stand_half_extents: tuple[float, float, float] = (0.2, 0.2, 0.225)

    human_base_pose: Pose = Pose(position=(1.0, 0.53, 0.39))
    human_joints: JointPositions = field(
        default_factory=lambda: [
            0.0,
            0.0,
            0.0,
            0.08726646,
            0.0,
            0.0,
            -1.57079633,
            0.0,
            0.0,
            0.0,
        ]
    )

    wheelchair_base_pose: Pose = Pose(position=(1.0, 0.5, -0.46))


class PyBulletHandoverSimulator:
    """A shared simulator used for both MDP and intake."""

    def __init__(
        self,
        scene_description: PyBulletHandoverSceneDescription,
        use_gui: bool = False,
        seed: int = 0,
    ) -> None:

        self._rng = np.random.default_rng(seed)
        self.scene_description = scene_description

        # Create the PyBullet client.
        if use_gui:
            self.physics_client_id = create_gui_connection(camera_yaw=0)
        else:
            self.physics_client_id = p.connect(p.DIRECT)

        # Create robot.
        robot = create_pybullet_robot(
            self.scene_description.robot_name,
            self.physics_client_id,
            base_pose=self.scene_description.robot_base_pose,
            control_mode="reset",
            home_joint_positions=self.scene_description.initial_joints,
        )
        assert isinstance(robot, FingeredSingleArmPyBulletRobot)
        robot.close_fingers()
        self.robot = robot

        # Create robot stand.
        self._robot_stand_id = create_pybullet_block(
            self.scene_description.robot_stand_rgba,
            half_extents=self.scene_description.robot_stand_half_extents,
            physics_client_id=self.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self._robot_stand_id,
            self.scene_description.robot_stand_pose.position,
            self.scene_description.robot_stand_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Create human.
        human_creation = HumanCreation(
            self.physics_client_id, np_random=self._rng, cloth=False
        )
        self.human = Human([], controllable=False)
        self.human.init(
            human_creation,
            static_human_base=True,
            impairment="none",
            gender="male",
            config=None,
            id=self.physics_client_id,
            np_random=self._rng,
        )
        p.resetBasePositionAndOrientation(
            self.human.body,
            self.scene_description.human_base_pose.position,
            self.scene_description.human_base_pose.orientation,
            physicsClientId=self.physics_client_id,
        )
        # Use some default joint positions from assistive gym first.
        joints_positions = [
            (self.human.j_right_elbow, -90),
            (self.human.j_left_elbow, -90),
            (self.human.j_right_hip_x, -90),
            (self.human.j_right_knee, 80),
            (self.human.j_left_hip_x, -90),
            (self.human.j_left_knee, 80),
            (self.human.j_head_x, 0.0),
            (self.human.j_head_y, 0.0),
            (self.human.j_head_z, 0.0),
        ]
        self.human.setup_joints(
            joints_positions, use_static_joints=True, reactive_force=None
        )
        # Now set arm joints using scene description.
        self.human.set_joint_angles(
            self.human.right_arm_joints, self.scene_description.human_joints
        )

        # Create wheelchair.
        furniture = Furniture()
        directory = Path(assistive_gym.envs.__file__).parent / "assets"
        assert directory.exists()
        furniture.init(
            "wheelchair",
            directory,
            self.physics_client_id,
            self._rng,
            wheelchair_mounted=False,
        )
        p.resetBasePositionAndOrientation(
            furniture.body,
            self.scene_description.wheelchair_base_pose.position,
            self.scene_description.wheelchair_base_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Placeholder for full range of motion model.
        self.rom_sphere_center = get_link_pose(
            self.human.body, self.human.right_wrist, self.physics_client_id
        ).position
        self.rom_sphere_radius = 0.25
        # Visualize.
        shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.rom_sphere_radius,
            rgbaColor=(1.0, 0.0, 0.0, 0.5),
            physicsClientId=self.physics_client_id,
        )
        collision_id = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=1e-6,
            physicsClientId=self.physics_client_id,
        )
        self._rom_viz_id = p.createMultiBody(
            baseMass=-1,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=shape_id,
            basePosition=self.rom_sphere_center,
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=self.physics_client_id,
        )

        # Uncomment for debug / development.
        # while True:
        #     p.stepSimulation(self.physics_client_id)

    def get_state(self) -> _HandoverState:
        """Get the underlying state from the simulator."""
        robot_base = self.robot.get_base_pose()
        robot_joints = self.robot.get_joint_positions()
        human_base = get_link_pose(self.human.body, -1, self.physics_client_id)
        human_joints = self.human.get_joint_angles(self.human.right_arm_joints)
        return _HandoverState(robot_base, robot_joints, human_base, human_joints)

    def set_state(self, state: _HandoverState) -> None:
        """Sync the simulator with the given state."""
        p.resetBasePositionAndOrientation(
            self.robot.robot_id,
            state.robot_base.position,
            state.robot_base.orientation,
            self.physics_client_id,
        )
        self.robot.set_joints(state.robot_joints)
        p.resetBasePositionAndOrientation(
            self.human.body,
            state.human_base.position,
            state.human_base.orientation,
            self.physics_client_id,
        )
        self.human.set_joint_angles(
            self.human.right_arm_joints,
            state.human_joints,
        )

    def step(self, action: JointPositions) -> None:
        """Advance the simulator given an action."""
        # Update the robot arm angles.
        current_joints = self.robot.get_joint_positions()
        # Only update the arm, assuming the first 7 entries are the arm.
        arm_joints = current_joints[:7]
        new_arm_joints = np.add(arm_joints, action)
        new_joints = list(current_joints)
        new_joints[:7] = new_arm_joints
        clipped_joints = np.clip(
            new_joints, self.robot.joint_lower_limits, self.robot.joint_upper_limits
        )
        self.robot.set_joints(clipped_joints.tolist())


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
        return gym.spaces.OneOf(
            (
                gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                EnumSpace([None]),
            )
        )

    def state_is_terminal(self, state: _HandoverState) -> bool:
        # Will be replaced by a real ROM check later.
        end_effector_pose = self._sim.robot.forward_kinematics(state.robot_joints)
        dist = np.sqrt(
            np.sum(
                np.subtract(end_effector_pose.position, self._sim.rom_sphere_center)
                ** 2
            )
        )
        return dist < self._sim.rom_sphere_radius

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
        # In the future, will actually randomize this.
        robot_base = self._sim.scene_description.robot_base_pose
        robot_joints = self._sim.scene_description.initial_joints
        human_base = self._sim.scene_description.human_base_pose
        human_joints = self._sim.scene_description.human_joints
        return _HandoverState(robot_base, robot_joints, human_base, human_joints)

    def get_transition_distribution(
        self, state: _HandoverState, action: _HandoverAction
    ) -> CategoricalDistribution:
        raise NotImplementedError("Sample transitions, don't enumerate them")

    def sample_next_state(
        self, state: _HandoverState, action: _HandoverAction, rng: np.random.Generator
    ) -> _HandoverState:
        if np.isclose(action[0], 1):
            return state  # will handle none case later
        joint_angle_delta = action[1]
        assert joint_angle_delta is not None
        # At the moment, this is actually deterministic.
        self._sim.set_state(state)
        self._sim.step(joint_angle_delta)
        return self._sim.get_state()

    def render_state(self, state: _HandoverState) -> Image:
        self._sim.set_state(state)
        target = get_link_pose(
            self._sim.human.body,
            self._sim.human.right_wrist,
            self._sim.physics_client_id,
        ).position
        return capture_image(self._sim.physics_client_id, camera_target=target)


@dataclass(frozen=True)
class _ROMReachableQuestion:
    """Ask the person to try to reach a certain position."""

    position: Pose3D  # in absolute coordinates

    def __lt__(self, other: Any) -> bool:
        return str(self) < str(other)


_HandoverIntakeObs: TypeAlias = bool  # whether or not reaching is successful
_HandoverIntakeAction: TypeAlias = NDArray[np.float32]  # _ROMReachableQuestion


class PyBulletHandoverIntakeProcess(
    IntakeProcess[_HandoverIntakeObs, _HandoverIntakeAction]
):
    """Intake process for the pybullet handover environment."""

    def __init__(self, horizon: int, sim: PyBulletHandoverSimulator) -> None:
        self._horizon = horizon
        self._sim = sim

    @property
    def observation_space(self) -> EnumSpace[_HandoverIntakeObs]:
        return EnumSpace([True, False])

    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)

    @property
    def horizon(self) -> int:
        return self._horizon

    def get_observation_distribution(
        self,
        action: _HandoverIntakeAction,
    ) -> CategoricalDistribution[_HandoverIntakeObs]:
        # Coming in next PR.
        return CategoricalDistribution({True: 0.5, False: 0.5})


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
