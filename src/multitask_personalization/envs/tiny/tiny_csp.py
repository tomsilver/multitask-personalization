"""CSP elements for the tiny environment."""

from typing import Any

import numpy as np
from gymnasium.spaces import Box

from multitask_personalization.envs.tiny.tiny_env import TinyAction, TinyState
from multitask_personalization.structs import (
    CSP,
    CSPConstraint,
    CSPPolicy,
    CSPSampler,
    CSPVariable,
    FunctionalCSPSampler,
    TrainableCSPConstraint,
)


class TinyUserConstraint(TrainableCSPConstraint[TinyState, TinyAction]):
    """User proximity preference for the tiny environment."""

    def __init__(
        self,
        position_var: CSPVariable,
        human_position: float,
        init_desired_distance: float = 1.0,
        distance_threshold: float = 1e-1,
    ) -> None:
        super().__init__(
            "user_preference",
            [position_var],
            self._position_close_enough,
        )
        self._human_position = human_position
        # Updated through learning.
        self._desired_distance = init_desired_distance
        self._distance_threshold = distance_threshold
        # Training data for learning.
        self._training_inputs: list[float] = []
        self._training_outputs: list[bool] = []

    def _position_close_enough(self, position: np.float_) -> bool:
        dist = abs(self._human_position - position)
        return bool(abs(dist - self._desired_distance) < self._distance_threshold)

    def learn_from_transition(
        self,
        obs: TinyState,
        act: TinyAction,
        next_obs: TinyState,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        # Only learn from cases where the robot triggered "done".
        if not np.isclose(act[0], 1):
            return
        assert act[1] is None
        # Check if the trigger was successful.
        label = reward > 0
        # Get the current distance.
        dist = abs(obs.robot - obs.human)
        # Update the training data.
        self._training_inputs.append(dist)
        self._training_outputs.append(label)
        # Update the constraint parameters.
        self._update_constraint_parameters()

    def _update_constraint_parameters(self) -> None:
        positive_dists: set[float] = set()
        for d, l in zip(self._training_inputs, self._training_outputs, strict=True):
            if l:
                positive_dists.add(d)
        if not positive_dists:
            return  # need to wait for data
        min_positive_dist = min(positive_dists)
        max_positive_dist = max(positive_dists)
        self._desired_distance = (min_positive_dist + max_positive_dist) / 2


class _TinyCSPPolicy(CSPPolicy[TinyState, TinyAction]):

    def __init__(
        self, csp: CSP, seed: int = 0, distance_threshold: float = 1e-1
    ) -> None:
        super().__init__(csp, seed)
        self._target_position: float | None = None
        self._distance_threshold = distance_threshold

    def reset(self, solution: dict[CSPVariable, Any]) -> None:
        super().reset(solution)
        self._target_position = self._get_value("position")

    def step(self, obs: TinyState) -> TinyAction:
        assert self._target_position is not None
        robot_position = obs.robot
        delta = np.clip(self._target_position - robot_position, -1, 1)
        if abs(delta) < 1e-6:
            return (1, None)
        return (0, delta)


def create_tiny_csp(
    human_position: float,
    seed: int = 0,
    distance_threshold: float = 1e-1,
) -> tuple[CSP, list[CSPSampler], CSPPolicy, dict[CSPVariable, Any]]:
    """Create a CSP for the tiny environment."""

    ################################ Variables ################################

    # Choose a position region to target.
    position = CSPVariable("position", Box(-np.inf, np.inf, shape=(), dtype=np.float_))
    variables = [position]

    ############################## Initialization #############################

    initialization = {
        position: 0.0,
    }

    ############################### Constraints ###############################

    # Create a user preference constraint.
    user_preference_constraint = TinyUserConstraint(
        position, human_position, distance_threshold=distance_threshold
    )
    constraints: list[CSPConstraint] = [user_preference_constraint]

    ################################### CSP ###################################

    csp = CSP(variables, constraints)

    ################################# Samplers ################################

    def _sample_position_fn(
        _: dict[CSPVariable, Any], rng: np.random.Generator
    ) -> dict[CSPVariable, Any]:
        sample = rng.normal(loc=human_position, scale=10.0)
        return {position: sample}

    position_sampler = FunctionalCSPSampler(_sample_position_fn, csp, {position})

    samplers: list[CSPSampler] = [position_sampler]

    ################################# Policy ##################################

    policy: CSPPolicy = _TinyCSPPolicy(
        csp, seed=seed, distance_threshold=distance_threshold
    )

    return csp, samplers, policy, initialization