"""An approach that generates and solves a CSP to make decisions."""

import logging
from typing import Any

import gymnasium as gym

from multitask_personalization.envs.pybullet.pybullet_csp import PyBulletCSPGenerator
from multitask_personalization.envs.pybullet.pybullet_env import (
    PyBulletEnv,
    PyBulletState,
)
from multitask_personalization.envs.pybullet.pybullet_task_spec import PyBulletTaskSpec
from multitask_personalization.envs.tiny.tiny_csp import TinyCSPGenerator
from multitask_personalization.envs.tiny.tiny_env import TinyState
from multitask_personalization.methods.approach import BaseApproach, _ActType, _ObsType
from multitask_personalization.rom.models import SphericalROMModel
from multitask_personalization.structs import (
    CSPGenerator,
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
        ensemble_explore_threshold: float = 0.1,
        max_motion_planning_candidates: int = 1,
    ):
        super().__init__(action_space, seed)
        self._explore_method = explore_method
        self._ensemble_explore_threshold = ensemble_explore_threshold
        self._current_policy: CSPPolicy | None = None
        self._current_sol: dict[CSPVariable, Any] | None = None
        self._currently_exploring = False
        self._csp_generator: CSPGenerator | None = None
        self._max_motion_planning_candidates = max_motion_planning_candidates

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
                    seed=self._seed,
                    explore_method=self._explore_method,
                    ensemble_explore_threshold=self._ensemble_explore_threshold,
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
                    ensemble_explore_threshold=self._ensemble_explore_threshold,
                    max_motion_planning_candidates=self._max_motion_planning_candidates,
                )
            else:
                raise NotImplementedError()
        self._recompute_policy(obs, explore=info["explore"])

    def _recompute_policy(self, obs: _ObsType, explore: bool = False) -> None:
        assert isinstance(self._csp_generator, CSPGenerator)
        csp, samplers, policy, initialization = self._csp_generator.generate(
            obs,
            explore=explore,
        )
        self._current_sol = solve_csp(
            csp,
            initialization,
            samplers,
            self._rng,
        )
        if self._current_sol is None:
            # For now we assume that exploration fails only due to pessimistic
            # learned constraints. So we just rerun with explore=True.
            # In the future, implement some fallback behavior in case this just
            # straight-up fails.
            assert not explore
            self._recompute_policy(obs, explore=True)
        else:
            self._current_policy = policy
            self._current_policy.reset(self._current_sol)
            self._currently_exploring = explore

    def _get_action(self) -> _ActType:
        assert self._last_observation is not None
        assert self._csp_generator is not None
        need_to_recompute_policy = False
        if self._current_policy is None:
            logging.info("Recomputing policy because of termination")
            need_to_recompute_policy = True
        # Check if the current solution to the CSP is still valid--it may not be
        # because the CSP constraints may have changed as a result of learning.
        # In this case, we trigger replanning by calling reset(). Note also we
        # need to regenerate the CSP because CSPs are not stateful.
        else:
            assert self._current_sol is not None
            csp, _, _, _ = self._csp_generator.generate(
                self._last_observation,
                explore=self._currently_exploring,
            )
            if not csp.check_solution(self._current_sol):
                logging.info("Recomputing policy because CSP violated online")
                need_to_recompute_policy = True
        if need_to_recompute_policy:
            self._recompute_policy(
                self._last_observation, explore=self._currently_exploring
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

    def get_episode_metrics(self) -> dict[str, float]:
        episode_metrics = super().get_episode_metrics()
        assert self._csp_generator is not None
        csp_metrics = self._csp_generator.get_metrics()
        assert not set(csp_metrics) & set(episode_metrics), "Metric name conflict"
        episode_metrics.update(csp_metrics)
        return episode_metrics
