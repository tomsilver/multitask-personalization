"""CSP elements for the tiny environment."""

from typing import Any

import numpy as np
import scipy.stats
from gymnasium.spaces import Box

from multitask_personalization.csp_generation import (
    CSPConstraintGenerator,
    CSPGenerator,
)
from multitask_personalization.envs.tiny.tiny_env import TinyAction, TinyState
from multitask_personalization.structs import (
    CSP,
    CSPConstraint,
    CSPCost,
    CSPPolicy,
    CSPSampler,
    CSPVariable,
    FunctionalCSPSampler,
    LogProbCSPConstraint,
)


class _TinyCSPPolicy(CSPPolicy[TinyState, TinyAction]):

    def __init__(
        self, csp: CSP, seed: int = 0, distance_threshold: float = 1e-1
    ) -> None:
        super().__init__(csp, seed)
        self._target_position: float | None = None
        self._speed: float | None = None
        self._distance_threshold = distance_threshold

    def reset(self, solution: dict[CSPVariable, Any]) -> None:
        super().reset(solution)
        self._target_position = self._get_value("position")
        self._speed = self._get_value("speed")

    def step(self, obs: TinyState) -> tuple[TinyAction, bool]:
        assert self._target_position is not None
        assert self._speed is not None
        robot_position = obs.robot
        delta = np.clip(self._target_position - robot_position, -1, 1)
        delta = self._speed * delta
        if abs(delta) < 1e-6:
            return (1, None), True
        return (0, delta), False


class _TinyDistanceConstraintGenerator(CSPConstraintGenerator[TinyState, TinyAction]):
    """Generates distance constraints for the TinyEnv."""

    def __init__(
        self,
        seed: int = 0,
        distance_threshold: float = 1e-1,
        init_desired_distance: float = 1.0,
        learning_rate: float = 1e-1,
    ) -> None:
        super().__init__(seed=seed)
        self._learning_rate = learning_rate
        self._distance_threshold = distance_threshold
        # Updated through learning.
        self._desired_distance = init_desired_distance
        # Training data for learning.
        self._training_inputs: list[float] = []
        self._training_outputs: list[bool] = []

    def generate(
        self,
        obs: TinyState,
        csp_vars: list[CSPVariable],
        constraint_name: str,
    ) -> CSPConstraint:

        assert len(csp_vars) == 1
        position_var = next(iter(csp_vars))

        def _position_logprob(position: np.float_) -> float:
            dist = abs(obs.human - position)
            return scipy.stats.norm.logpdf(dist, loc=self._desired_distance)

        threshold = scipy.stats.norm.logpdf(self._distance_threshold)
        user_preference_constraint = LogProbCSPConstraint(
            constraint_name,
            [position_var],
            _position_logprob,
            threshold=threshold,
        )
        return user_preference_constraint

    def learn_from_transition(
        self,
        obs: TinyState,
        act: TinyAction,
        next_obs: TinyState,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        # Only learn from cases where the robot triggered "done".
        if not np.isclose(act[0], 1):
            return
        assert act[1] is None
        # Check if the trigger was successful.
        label = info["success"]
        # Get the current distance.
        dist = abs(obs.robot - obs.human)
        # Update the training data.
        self._training_inputs.append(dist)
        self._training_outputs.append(label)
        # Update the constraint parameters.
        self._update_constraint_parameters()

    def get_metrics(self) -> dict[str, float]:
        return {
            "tiny_user_proximity_learned_distance": self._desired_distance,
        }

    def _update_constraint_parameters(self) -> None:
        positive_dists: set[float] = set()
        for d, l in zip(self._training_inputs, self._training_outputs, strict=True):
            if l:
                positive_dists.add(d)
        if not positive_dists:
            return  # need to wait for data
        min_positive_dist = min(positive_dists)
        max_positive_dist = max(positive_dists)
        center = (min_positive_dist + max_positive_dist) / 2
        delta = center - self._desired_distance
        self._desired_distance += self._learning_rate * delta


class TinyCSPGenerator(CSPGenerator[TinyState, TinyAction]):
    """Create a CSP for the tiny environment."""

    def __init__(
        self,
        distance_threshold: float = 1e-1,
        init_desired_distance: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._distance_threshold = distance_threshold
        self._distance_constraint_generator = _TinyDistanceConstraintGenerator(
            seed=self._seed,
            distance_threshold=distance_threshold,
            init_desired_distance=init_desired_distance,
        )

    def _generate_variables(
        self,
        obs: TinyState,
    ) -> tuple[list[CSPVariable], dict[CSPVariable, Any]]:
        # Choose a position to target and a speed to move.
        position = CSPVariable(
            "position", Box(-np.inf, np.inf, shape=(), dtype=np.float_)
        )
        speed = CSPVariable("speed", Box(0, 1, shape=(), dtype=np.float_))
        variables = [position, speed]
        initialization = {
            position: self._rng.uniform(-10, 10),
            speed: self._rng.uniform(0, 1),
        }
        return variables, initialization

    def _generate_personal_constraints(
        self,
        obs: TinyState,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:
        position, _ = variables
        user_preference_constraint = self._distance_constraint_generator.generate(
            obs, [position], "user_preference"
        )
        return [user_preference_constraint]

    def _generate_nonpersonal_constraints(
        self,
        obs: TinyState,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:
        return []

    def _generate_exploit_cost(
        self,
        obs: TinyState,
        variables: list[CSPVariable],
    ) -> CSPCost | None:

        _, speed = variables

        def _speed_cost_fn(x: np.float_) -> float:
            """Move as fast as possible."""
            return 1.0 - float(x)

        return CSPCost("maximize-speed", [speed], _speed_cost_fn)

    def _generate_samplers(
        self,
        obs: TinyState,
        csp: CSP,
    ) -> list[CSPSampler]:

        position, speed = csp.variables
        human_position = obs.human

        def _sample_position_fn(
            _: dict[CSPVariable, Any], rng: np.random.Generator
        ) -> dict[CSPVariable, Any]:
            sample = rng.normal(loc=human_position, scale=10.0)
            return {position: sample}

        def _sample_speed_fn(
            _: dict[CSPVariable, Any], rng: np.random.Generator
        ) -> dict[CSPVariable, Any]:
            sample = rng.uniform(0, 1)
            return {speed: sample}

        position_sampler = FunctionalCSPSampler(_sample_position_fn, csp, {position})
        speed_sampler = FunctionalCSPSampler(_sample_speed_fn, csp, {speed})

        return [position_sampler, speed_sampler]

    def _generate_policy(
        self,
        obs: TinyState,
        csp: CSP,
    ) -> CSPPolicy:
        return _TinyCSPPolicy(
            csp, seed=self._seed, distance_threshold=self._distance_threshold
        )

    def learn_from_transition(
        self,
        obs: TinyState,
        act: TinyAction,
        next_obs: TinyState,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        self._distance_constraint_generator.learn_from_transition(
            obs, act, next_obs, done, info
        )

    def get_metrics(self) -> dict[str, float]:
        return self._distance_constraint_generator.get_metrics()
