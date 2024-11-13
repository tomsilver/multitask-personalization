"""CSP elements for the tiny environment."""

from typing import Any

import numpy as np
from gymnasium.spaces import Box

from multitask_personalization.envs.tiny.tiny_env import TinyAction, TinyState
from multitask_personalization.structs import (
    CSP,
    CSPConstraint,
    CSPConstraintGenerator,
    CSPGenerator,
    CSPPolicy,
    CSPSampler,
    CSPVariable,
    EnsembleCSPConstraintGenerator,
    FunctionalCSPSampler,
)


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

    def step(self, obs: TinyState) -> tuple[TinyAction, bool]:
        assert self._target_position is not None
        robot_position = obs.robot
        delta = np.clip(self._target_position - robot_position, -1, 1)
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
        neighborhood: float = 0.0,
    ) -> CSPConstraint:

        assert len(csp_vars) == 1
        position_var = next(iter(csp_vars))

        def _position_close_enough(position: np.float_) -> bool:
            dist = abs(obs.human - position)
            return bool(
                abs(dist - self._desired_distance)
                < self._distance_threshold + neighborhood
            )

        user_preference_constraint = CSPConstraint(
            constraint_name, [position_var], _position_close_enough
        )
        return user_preference_constraint

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
        seed: int = 0,
        explore_method: str = "nothing-personal",
        ensemble_explore_threshold: float = 1e-1,
        ensemble_explore_members: int = 5,
        neighborhood_explore_max_radius: float = 10.0,
        neighborhood_explore_radius_decay: float = 0.99,
        distance_threshold: float = 1e-1,
        init_desired_distance: float = 1.0,
    ) -> None:
        super().__init__(
            seed=seed,
            explore_method=explore_method,
            ensemble_explore_threshold=ensemble_explore_threshold,
            ensemble_explore_members=ensemble_explore_members,
            neighborhood_explore_max_radius=neighborhood_explore_max_radius,
            neighborhood_explore_radius_decay=neighborhood_explore_radius_decay,
        )
        self._distance_threshold = distance_threshold
        self._num_generations = 0
        if explore_method == "ensemble":
            members: list[CSPConstraintGenerator] = []
            for idx in range(self._ensemble_explore_members):
                init = max(0, init_desired_distance + self._rng.normal(0, 10))
                member = _TinyDistanceConstraintGenerator(
                    seed=(seed + idx),
                    distance_threshold=distance_threshold,
                    init_desired_distance=init,
                )
                members.append(member)
            self._distance_constraint_generator: CSPConstraintGenerator = (
                EnsembleCSPConstraintGenerator(members)
            )
        else:
            self._distance_constraint_generator = _TinyDistanceConstraintGenerator(
                seed=seed,
                distance_threshold=distance_threshold,
                init_desired_distance=init_desired_distance,
            )

    def generate(
        self,
        obs: TinyState,
        explore: bool = False,
    ) -> tuple[
        CSP, list[CSPSampler], CSPPolicy[TinyState, TinyAction], dict[CSPVariable, Any]
    ]:

        human_position = obs.human

        ################################ Variables ################################

        # Choose a position to target.
        position = CSPVariable(
            "position", Box(-np.inf, np.inf, shape=(), dtype=np.float_)
        )
        variables = [position]

        ############################## Initialization #############################

        initialization = {
            position: 0.0,
        }

        ############################### Constraints ###############################

        constraints: list[CSPConstraint] = []

        if not explore:
            user_preference_constraint = self._distance_constraint_generator.generate(
                obs, [position], "user_preference"
            )
            constraints.append(user_preference_constraint)
        elif self._explore_method == "ensemble":
            assert isinstance(
                self._distance_constraint_generator, EnsembleCSPConstraintGenerator
            )
            user_preference_constraint = self._distance_constraint_generator.generate(
                obs,
                [position],
                "user_preference",
                member_classification_threshold=self._ensemble_explore_threshold,
            )
            constraints.append(user_preference_constraint)
        elif self._explore_method == "neighborhood":
            radius = self._neighborhood_explore_max_radius * (
                self._neighborhood_explore_radius_decay**self._num_generations
            )
            user_preference_constraint = self._distance_constraint_generator.generate(
                obs,
                [position],
                "user_preference",
                neighborhood=radius,
            )
            constraints.append(user_preference_constraint)
        else:
            assert self._explore_method == "nothing-personal"

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
            csp, seed=self._seed, distance_threshold=self._distance_threshold
        )

        self._num_generations += 1

        return csp, samplers, policy, initialization

    def learn_from_transition(
        self,
        obs: TinyState,
        act: TinyAction,
        next_obs: TinyState,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        self._distance_constraint_generator.learn_from_transition(
            obs, act, next_obs, reward, done, info
        )

    def get_metrics(self) -> dict[str, float]:
        return self._distance_constraint_generator.get_metrics()
