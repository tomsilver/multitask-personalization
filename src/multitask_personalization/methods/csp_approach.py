"""An approach that generates and solves a CSP to make decisions."""

import logging
from typing import Any

import gymnasium as gym

from multitask_personalization.csp_generation import CSPGenerator
from multitask_personalization.envs.pybullet.pybullet_csp import PyBulletCSPGenerator
from multitask_personalization.envs.pybullet.pybullet_env import (
    PyBulletEnv,
    PyBulletState,
)
from multitask_personalization.envs.pybullet.pybullet_task_spec import PyBulletTaskSpec
from multitask_personalization.envs.tiny.tiny_csp import TinyCSPGenerator
from multitask_personalization.envs.tiny.tiny_env import TinyState
from multitask_personalization.methods.approach import (
    ApproachFailure,
    BaseApproach,
    _ActType,
    _ObsType,
)
from multitask_personalization.rom.models import SphericalROMModel
from multitask_personalization.structs import (
    CSPPolicy,
    CSPVariable,
)
from multitask_personalization.utils import solve_csp


class CSPApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that generates and solves a CSP to make decisions."""

    def __init__(
        self,
        action_space: gym.spaces.Space[_ActType],
        seed: int,
        explore_method: str = "nothing-personal",
        max_motion_planning_candidates: int = 1,
        csp_min_num_satisfying_solutions: int = 50,
        show_csp_progress_bar: bool = True,
    ):
        super().__init__(action_space, seed)
        self._current_policy: CSPPolicy | None = None
        self._current_sol: dict[CSPVariable, Any] | None = None
        self._csp_generator: CSPGenerator | None = None
        self._explore_method = explore_method
        self._max_motion_planning_candidates = max_motion_planning_candidates
        self._csp_min_num_satisfying_solutions = csp_min_num_satisfying_solutions
        self._show_csp_progress_bar = show_csp_progress_bar

    def reset(
        self,
        obs: _ObsType,
        info: dict[str, Any],
    ) -> None:
        super().reset(obs, info)
        if self._csp_generator is None:
            # At the moment, this part is extremely environment-specific.
            # We will refactor this in a future PR.
            if isinstance(obs, TinyState):
                self._csp_generator = TinyCSPGenerator(
                    seed=self._seed, explore_method=self._explore_method
                )
            elif isinstance(obs, PyBulletState):
                task_spec = info["task_spec"]
                assert isinstance(task_spec, PyBulletTaskSpec)
                sim = PyBulletEnv(task_spec, seed=self._seed, use_gui=False)
                rom_model = SphericalROMModel(task_spec.human_spec, self._seed)
                self._csp_generator = PyBulletCSPGenerator(
                    sim,
                    rom_model,
                    seed=self._seed,
                    explore_method=self._explore_method,
                    max_motion_planning_candidates=self._max_motion_planning_candidates,
                )
            else:
                raise NotImplementedError()
        self._recompute_policy(obs, user_allows_explore=info["user_allows_explore"])

    def _recompute_policy(
        self, obs: _ObsType, user_allows_explore: bool = False
    ) -> None:
        assert isinstance(self._csp_generator, CSPGenerator)
        csp, samplers, policy, initialization = self._csp_generator.generate(
            obs,
            user_allows_explore=user_allows_explore,
        )
        self._current_sol = solve_csp(
            csp,
            initialization,
            samplers,
            self._rng,
            min_num_satisfying_solutions=self._csp_min_num_satisfying_solutions,
            show_progress_bar=self._show_csp_progress_bar,
        )
        if self._current_sol is None:
            raise ApproachFailure("No solution found for generated CSP")
        self._current_policy = policy
        self._current_policy.reset(self._current_sol)

    def _get_action(self) -> _ActType:
        assert self._last_observation is not None
        assert self._last_info is not None
        assert self._csp_generator is not None
        if self._current_policy is None:
            logging.info("Recomputing policy because of termination")
            self._recompute_policy(
                self._last_observation,
                user_allows_explore=self._last_info["user_allows_explore"],
            )
        assert self._current_policy is not None
        action, done = self._current_policy.step(self._last_observation)
        if done:
            self._current_policy = None
        return action

    def _learn_from_transition(
        self,
        obs: _ObsType,
        act: _ActType,
        next_obs: _ObsType,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        assert self._csp_generator is not None
        self._csp_generator.learn_from_transition(
            obs, act, next_obs, reward, done, info
        )

    def get_step_metrics(self) -> dict[str, float]:
        step_metrics = super().get_step_metrics()
        assert self._csp_generator is not None
        csp_metrics = self._csp_generator.get_metrics()
        assert not set(csp_metrics) & set(step_metrics), "Metric name conflict"
        step_metrics.update(csp_metrics)
        return step_metrics
