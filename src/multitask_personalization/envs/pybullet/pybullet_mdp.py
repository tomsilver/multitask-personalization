"""The MDP for the pybullet environment."""

from __future__ import annotations

from functools import cached_property

import gymnasium as gym
import numpy as np
from pybullet_helpers.camera import capture_image
from pybullet_helpers.link import get_link_pose
from tomsutils.spaces import EnumSpace

from multitask_personalization.envs.mdp import MDP
from multitask_personalization.envs.pybullet.pybullet_sim import (
    PyBulletHandoverSimulator,
)
from multitask_personalization.envs.pybullet.pybullet_structs import (
    _GripperAction,
    _HandoverAction,
    _HandoverState,
)
from multitask_personalization.structs import (
    CategoricalDistribution,
    Image,
)


class PyBulletHandoverMDP(MDP[_HandoverState, _HandoverAction]):
    """A handover environment implemented in PyBullet."""

    def __init__(
        self,
        sim: PyBulletHandoverSimulator,
    ) -> None:
        self._sim = sim
        self._terminal_state_padding = 1e-2

    @cached_property
    def state_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            -np.inf, np.inf, shape=(_HandoverState.get_dimension(),), dtype=np.float32
        )

    @cached_property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.OneOf(
            (
                gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                EnumSpace([_GripperAction.OPEN, _GripperAction.CLOSE]),
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
        return dist < self._sim.rom_sphere_radius + self._terminal_state_padding

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
        object_pose = self._sim.scene_description.object_pose
        grasp_transform = None
        return _HandoverState(
            robot_base,
            robot_joints,
            human_base,
            human_joints,
            object_pose,
            grasp_transform,
        )

    def get_transition_distribution(
        self, state: _HandoverState, action: _HandoverAction
    ) -> CategoricalDistribution:
        raise NotImplementedError("Sample transitions, don't enumerate them")

    def sample_next_state(
        self, state: _HandoverState, action: _HandoverAction, rng: np.random.Generator
    ) -> _HandoverState:
        self._sim.set_state(state)
        self._sim.step(action)
        return self._sim.get_state()

    def render_state(self, state: _HandoverState) -> Image:
        self._sim.set_state(state)
        target = get_link_pose(
            self._sim.human.body,
            self._sim.human.right_wrist,
            self._sim.physics_client_id,
        ).position
        return capture_image(
            self._sim.physics_client_id,
            camera_target=target,
            camera_distance=self._sim.scene_description.camera_distance,
        )
