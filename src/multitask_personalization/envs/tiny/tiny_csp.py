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
        init_distance_threshold: float = 100.0,
    ) -> None:
        super().__init__("user_preference", [position_var], self._position_close_enough)
        self._human_position = human_position
        # Updated through learning.
        self._desired_distance = init_desired_distance
        self._distance_threshold = init_distance_threshold
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
        label = done
        # Get the current distance.
        dist = abs(obs.robot - obs.human)
        # Update the training data.
        self._training_inputs.append(dist)
        self._training_outputs.append(label)
        # Update the constraint parameters.
        self._update_constraint_parameters()

    def _update_constraint_parameters(self) -> None:
        positive_dists: list[float] = []
        negative_dists: list[float] = []
        for d, l in zip(self._training_inputs, self._training_outputs, strict=True):
            if l:
                positive_dists.append(d)
            else:
                negative_dists.append(d)
        if not positive_dists or not negative_dists:
            return  # need to wait for data
        min_positive_dist = min(positive_dists)
        max_positive_dist = max(positive_dists)
        self._desired_distance = (min_positive_dist + max_positive_dist) / 2
        # Keep the threshold optimistic... otherwise, the agent will fail to
        # optimistically explore after it has collected a small amount of data.
        negs_above = [d for d in negative_dists if d > self._desired_distance]
        negs_below = [d for d in negative_dists if d < self._desired_distance]
        if not negs_above or not negs_below:
            return  # wait to update threshold
        self._distance_threshold = min(
            min(negs_above) - self._desired_distance,
            self._desired_distance - max(negs_below),
        )


class _TinyCSPPolicy(CSPPolicy[TinyState, TinyAction]):

    def step(self, obs: TinyState) -> TinyAction:
        assert self._current_solution is not None
        assert len(self._current_solution) == 1
        target_position = next(iter(self._current_solution.values()))
        robot_position = obs.robot
        delta = np.clip(target_position - robot_position, -1, 1)
        if abs(delta) < 1e-6:
            return (1, None)
        return (0, delta)


def create_tiny_csp(
    human_position: float,
    seed: int = 0,
) -> tuple[CSP, list[CSPSampler], CSPPolicy, dict[CSPVariable, Any]]:
    """Create a CSP for the tiny environment."""

    ################################ Variables ################################

    # Choose a position to target.
    position = CSPVariable("position", Box(-np.inf, np.inf, shape=(), dtype=np.float_))
    variables = [position]

    ############################## Initialization #############################

    initialization = {
        position: 0.0,
    }

    ############################### Constraints ###############################

    # Create a user preference constraint.
    user_preference_constraint = TinyUserConstraint(position, human_position)
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

    policy: CSPPolicy = _TinyCSPPolicy(csp, seed=seed)

    return csp, samplers, policy, initialization
