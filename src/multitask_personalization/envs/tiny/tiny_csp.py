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
)


class _TinyCSPPolicy(CSPPolicy[TinyState, TinyAction]):

    def step(self, obs: TinyState) -> TinyAction:
        assert self._current_solution is not None
        assert len(self._current_solution) == 1
        target_position = next(iter(self._current_solution.values()))
        robot_position = obs.robot
        delta = np.clip(target_position - robot_position, -1, 1)
        if delta < 1e-6:
            return (1, None)
        return (0, delta)


def create_tiny_csp(
    human_position: float,
    desired_distance: float,
    distance_threshold: float,
    seed: int = 0,
) -> tuple[CSP, list[CSPSampler], CSPPolicy, dict[CSPVariable, Any]]:
    """Create a CSP for the tiny environment."""

    ################################ Variables ################################

    # Choose a position to target.
    position = CSPVariable("position", Box(-np.pi, np.pi, dtype=np.float_))
    variables = [position]

    ############################## Initialization #############################

    initialization = {
        position: 0.0,
    }

    ############################### Constraints ###############################

    # Create a user preference constraint.
    def _user_preference(position: np.float_) -> bool:
        dist = abs(human_position - position)
        return bool(abs(dist - desired_distance) < distance_threshold)

    user_preference_constraint = CSPConstraint(
        "user_preference",
        [position],
        _user_preference,
    )

    constraints = [
        user_preference_constraint,
    ]

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
