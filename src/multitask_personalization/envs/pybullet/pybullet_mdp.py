"""The MDP for the pybullet environment."""

from __future__ import annotations

from functools import cached_property

import gymnasium as gym
import numpy as np
from pybullet_helpers.camera import capture_image
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.link import get_link_pose
from tomsutils.spaces import EnumSpace

from multitask_personalization.envs.mdp import MDP
from multitask_personalization.envs.pybullet.pybullet_sim import (
    PyBulletSimulator,
)
from multitask_personalization.envs.pybullet.pybullet_structs import (
    _GripperAction,
    _PyBulletAction,
    _PyBulletState,
)
from multitask_personalization.structs import (
    CategoricalDistribution,
    Image,
)


class PyBulletMDP(MDP[_PyBulletState, _PyBulletAction]):
    """An environment implemented in PyBullet."""

    def __init__(
        self,
        sim: PyBulletSimulator,
    ) -> None:
        self._sim = sim
        self._terminal_state_padding = 1e-2

    @cached_property
    def state_space(self) -> gym.spaces.Box:
        raise NotImplementedError("TODO")

    @cached_property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.OneOf(
            (
                gym.spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32),
                EnumSpace([_GripperAction.OPEN, _GripperAction.CLOSE]),
                EnumSpace([None]),
            )
        )

    def state_is_terminal(self, state: _PyBulletState) -> bool:
        # Will be replaced by a real ROM check later.
        if "hand over" in self._sim.task_spec.task_objective:
            end_effector_pose = self._sim.robot.forward_kinematics(state.robot_joints)
            dist = np.sqrt(
                np.sum(
                    np.subtract(end_effector_pose.position, self._sim.rom_sphere_center)
                    ** 2
                )
            )
            return dist < self._sim.rom_sphere_radius + self._terminal_state_padding
        assert self._sim.task_spec.task_objective == "place books on tray"
        if self._sim.current_grasp_transform:
            return False
        for book_id in self._sim.book_ids:
            if not check_body_collisions(
                book_id,
                self._sim.tray_id,
                self._sim.physics_client_id,
                distance_threshold=1e-3,
            ):
                return False
        return True

    def get_reward(
        self, state: _PyBulletState, action: _PyBulletAction, next_state: _PyBulletState
    ) -> float:
        if self.state_is_terminal(next_state):
            return 1.0
        return 0.0

    def get_initial_state_distribution(
        self,
    ) -> CategoricalDistribution:
        raise NotImplementedError("Initial state distribution too large")

    def sample_initial_state(self, rng: np.random.Generator) -> _PyBulletState:
        # Randomization actually happens in the task spec.
        robot_base = self._sim.task_spec.robot_base_pose
        robot_joints = self._sim.task_spec.initial_joints
        human_base = self._sim.task_spec.human_base_pose
        human_joints = self._sim.task_spec.human_joints
        object_pose = self._sim.task_spec.object_pose
        book_poses = self._sim.task_spec.book_poses
        grasp_transform = None
        return _PyBulletState(
            robot_base,
            robot_joints,
            human_base,
            human_joints,
            object_pose,
            list(book_poses),
            grasp_transform,
        )

    def get_transition_distribution(
        self, state: _PyBulletState, action: _PyBulletAction
    ) -> CategoricalDistribution:
        raise NotImplementedError("Sample transitions, don't enumerate them")

    def sample_next_state(
        self, state: _PyBulletState, action: _PyBulletAction, rng: np.random.Generator
    ) -> _PyBulletState:
        self._sim.set_state(state)
        self._sim.step(action)
        return self._sim.get_state()

    def render_state(self, state: _PyBulletState) -> Image:
        self._sim.set_state(state)
        target = get_link_pose(
            self._sim.human.body,
            self._sim.human.right_wrist,
            self._sim.physics_client_id,
        ).position
        return capture_image(
            self._sim.physics_client_id,
            camera_target=target,
            camera_distance=self._sim.task_spec.camera_distance,
        )
