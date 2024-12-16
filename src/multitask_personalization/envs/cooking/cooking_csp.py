"""CSP generation for cooking environment."""

from multitask_personalization.csp_generation import CSPGenerator
from multitask_personalization.structs import (
    CSP,
    CSPConstraint,
    CSPCost,
    CSPPolicy,
    CSPSampler,
    CSPVariable,
    FunctionalCSPConstraint,
    FunctionalCSPSampler,
)
from multitask_personalization.envs.cooking.cooking_env import CookingState, CookingAction
from pathlib import Path
from typing import Any
from gymnasium.spaces import Box

class CookingCSPGenerator(CSPGenerator[CookingState, CookingAction]):
    """Create a CSP for the cooking environment."""

    def save(self, model_dir: Path) -> None:
        # Not yet learning anything.
        pass

    def load(self, model_dir: Path) -> None:
        # Not yet learning anything.
        pass

    def _generate_variables(
        self,
        obs: CookingState,
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
        obs: CookingState,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:
        position, _ = variables
        user_preference_constraint = self._distance_constraint_generator.generate(
            obs, [position], "user_preference"
        )
        return [user_preference_constraint]

    def _generate_nonpersonal_constraints(
        self,
        obs: CookingState,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:
        return []

    def _generate_exploit_cost(
        self,
        obs: CookingState,
        variables: list[CSPVariable],
    ) -> CSPCost | None:

        _, speed = variables

        def _speed_cost_fn(x: np.float_) -> float:
            """Move as fast as possible."""
            return 1.0 - float(x)

        return CSPCost("maximize-speed", [speed], _speed_cost_fn)

    def _generate_samplers(
        self,
        obs: CookingState,
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
        obs: CookingState,
        csp: CSP,
    ) -> CSPPolicy:
        return _TinyCSPPolicy(csp, seed=self._seed)

    def observe_transition(
        self,
        obs: CookingState,
        act: CookingAction,
        next_obs: CookingState,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        if not self._disable_learning:
            self._distance_constraint_generator.learn_from_transition(
                obs, act, next_obs, done, info
            )

    def get_metrics(self) -> dict[str, float]:
        return self._distance_constraint_generator.get_metrics()
