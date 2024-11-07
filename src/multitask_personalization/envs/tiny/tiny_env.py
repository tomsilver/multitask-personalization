"""A tiny environment from rapid development and testing."""

from dataclasses import dataclass
from typing import Any, TypeAlias

import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame
from tomsutils.spaces import EnumSpace


@dataclass(frozen=True)
class TinyState:
    """A state in the TinyEnv."""

    robot: float
    human: float


TinyAction: TypeAlias = tuple[int, float | None]  # delta move or declare done


@dataclass(frozen=True)
class TinyHiddenSpec:
    """A hidden specification of human user preferences."""

    desired_distance: float
    distance_threshold: float


class TinyEnv(gym.Env[TinyState, TinyAction]):
    """A tiny environment from rapid development and testing.

    A robot and a human are on a 1D line with randomly initialized
    location. The human has a hidden preference about how close the
    robot should be when it executes a "done" action.

    Actions are bounded delta movements for the robot.
    """

    def __init__(
        self,
        hidden_spec: TinyHiddenSpec | None = None,
        seed: int = 0,
        allow_explore_switch_prob: float = 5e-2,
    ) -> None:

        self._rng = np.random.default_rng(seed)
        self._hidden_spec = hidden_spec
        self._allow_explore_switch_prob = allow_explore_switch_prob

        self.action_space = gym.spaces.OneOf(
            (
                gym.spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32),
                EnumSpace([None]),
            )
        )

        # Reset in reset().
        self._robot_position = -1.0
        self._human_position = 1.0
        self._user_allows_explore = True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[TinyState, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        # Randomly reset the positions of the human and robot.
        self._robot_position = self._rng.uniform(-10.0, 10.0)
        self._reset_human()
        self._user_allows_explore = True
        return self._get_state(), self._get_info()

    def _reset_human(self) -> None:
        desired_distance = self._hidden_spec.desired_distance
        threshold = self._hidden_spec.distance_threshold
        while True:
            human_position = self._rng.uniform(-10.0 + (desired_distance + threshold), 10.0 - (desired_distance + threshold))
            dist = abs(self._robot_position - human_position)
            assert self._hidden_spec is not None
            if (
                dist
                >= desired_distance
                + threshold
            ):
                break
        self._human_position = human_position

    def step(
        self, action: TinyAction
    ) -> tuple[TinyState, float, bool, bool, dict[str, Any]]:
        if self._rng.uniform() < self._allow_explore_switch_prob:
            self._user_allows_explore = not self._user_allows_explore

        assert self.action_space.contains(action)
        if np.isclose(action[0], 1):
            reward = self._get_reward(robot_indicated_done=True)
        else:
            assert np.isclose(action[0], 0)
            delta_action = action[1]
            assert delta_action is not None
            self._robot_position += float(delta_action)
            reward = self._get_reward(robot_indicated_done=False)
        # Move the human if the robot succeeded.
        if reward > 0:
            self._reset_human()
        return self._get_state(), reward, False, False, self._get_info()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        raise NotImplementedError

    def _get_state(self) -> TinyState:
        return TinyState(self._robot_position, self._human_position)

    def _get_reward(self, robot_indicated_done: bool = False) -> float:
        if self._hidden_spec is None:
            raise NotImplementedError("Should not call step() in sim")
        # Robot needs to indicate done.
        if not robot_indicated_done:
            return 0.0
        dist = abs(self._robot_position - self._human_position)
        desired_dist = self._hidden_spec.desired_distance
        if abs(dist - desired_dist) < self._hidden_spec.distance_threshold:
            # Success!
            return 1.0
        # Penalize if not close enough to human.
        return -1.0

    def _get_info(self) -> dict[str, Any]:
        return {
            "user_allows_explore": self._user_allows_explore,
        }
