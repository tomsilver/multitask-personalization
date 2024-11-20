"""CSP elements for the tiny environment."""

import pickle as pkl
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray
from sklearn.neighbors import RadiusNeighborsClassifier

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
        self,
        csp: CSP,
        seed: int = 0,
    ) -> None:
        super().__init__(csp, seed)
        self._target_position: float | None = None
        self._speed: float | None = None
        self._terminated = False

    def reset(self, solution: dict[CSPVariable, Any]) -> None:
        super().reset(solution)
        self._target_position = self._get_value("position")
        self._speed = self._get_value("speed")
        self._terminated = False

    def step(self, obs: TinyState) -> TinyAction:
        assert self._target_position is not None
        assert self._speed is not None
        robot_position = obs.robot
        delta = self._speed * np.clip(self._target_position - robot_position, -1, 1)
        if abs(delta) < 1e-6:
            self._terminated = True
            return (1, None)
        return (0, delta)

    def check_termination(self, obs: TinyState) -> bool:
        return self._terminated


class _TinyDistanceConstraintGenerator(CSPConstraintGenerator[TinyState, TinyAction]):
    """Generates distance constraints for the TinyEnv."""

    def __init__(
        self,
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed)
        # Updated through learning.
        self._classifier: RadiusNeighborsClassifier | None = None
        # Training data for learning.
        self._training_inputs: list[NDArray] = []
        self._training_outputs: list[bool] = []

    def save(self, model_dir: Path) -> None:
        """Save classifier."""
        outfile = model_dir / "tiny_distance_constraint_classifier.json"
        with open(outfile, "wb") as f:
            pkl.dump(self._classifier, f)

    def load(self, model_dir: Path) -> None:
        """Load parameters."""
        outfile = model_dir / "tiny_distance_constraint_classifier.json"
        with open(outfile, "rb") as f:
            self._classifier = pkl.load(f)

    def generate(
        self,
        obs: TinyState,
        csp_vars: list[CSPVariable],
        constraint_name: str,
    ) -> CSPConstraint:

        assert len(csp_vars) == 1
        position_var = next(iter(csp_vars))

        def _position_logprob(position: np.float_) -> float:
            if self._classifier is None:
                return 0.0
            dist = abs(obs.human - position)
            x = self._featurize_input(dist)
            y = np.log(self._classifier.predict_proba([x])[0][1])
            return y

        user_preference_constraint = LogProbCSPConstraint(
            constraint_name,
            [position_var],
            _position_logprob,
            threshold=np.log(0.5),
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
        label = info["user_satisfaction"] > 0
        # Get the current distance.
        dist = abs(obs.robot - obs.human)
        # Update the training data.
        self._training_inputs.append(self._featurize_input(dist))
        self._training_outputs.append(label)
        # Update the constraint parameters.
        self._update_constraint_parameters()

    def get_metrics(self) -> dict[str, float]:
        return {}

    def _update_constraint_parameters(self) -> None:
        # Wait until we've seen both positive and negative examples to learn.
        if len(set(self._training_outputs)) < 2:
            return
        # Train a classifier.
        self._classifier = RadiusNeighborsClassifier(radius=1000.0, weights="distance")
        self._classifier.fit(self._training_inputs, self._training_outputs)

    def _featurize_input(self, dist: np.floating | float) -> NDArray:
        return np.array([dist])


class TinyCSPGenerator(CSPGenerator[TinyState, TinyAction]):
    """Create a CSP for the tiny environment."""

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._distance_constraint_generator = _TinyDistanceConstraintGenerator(
            seed=self._seed,
        )

    def save(self, model_dir: Path) -> None:
        self._distance_constraint_generator.save(model_dir)

    def load(self, model_dir: Path) -> None:
        self._distance_constraint_generator.load(model_dir)

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
        return _TinyCSPPolicy(csp, seed=self._seed)

    def observe_transition(
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
