"""An approach that generates and solves a CSP to make decisions."""

import logging
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from multitask_personalization.csp_generation import CSPGenerator
from multitask_personalization.envs.pybullet.pybullet_csp import PyBulletCSPGenerator
from multitask_personalization.envs.pybullet.pybullet_env import (
    PyBulletEnv,
    PyBulletState,
)
from multitask_personalization.envs.pybullet.pybullet_scene_spec import (
    PyBulletSceneSpec,
)
from multitask_personalization.envs.tiny.tiny_csp import TinyCSPGenerator
from multitask_personalization.envs.tiny.tiny_env import TinyState
from multitask_personalization.methods.approach import (
    ApproachFailure,
    BaseApproach,
    _ActType,
    _ObsType,
)
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
        csp_generator: CSPGenerator,
        max_motion_planning_candidates: int = 1,
        csp_min_num_satisfying_solutions: int = 50,
        show_csp_progress_bar: bool = True,
    ):
        super().__init__(action_space, seed)
        self._csp_generator = csp_generator
        self._current_policy: CSPPolicy | None = None
        self._current_sol: dict[CSPVariable, Any] | None = None
        self._max_motion_planning_candidates = max_motion_planning_candidates
        self._csp_min_num_satisfying_solutions = csp_min_num_satisfying_solutions
        self._show_csp_progress_bar = show_csp_progress_bar

    def reset(
        self,
        obs: _ObsType,
        info: dict[str, Any],
    ) -> None:
        super().reset(obs, info)
        # TODO remove!!!!
        # if self._csp_generator is None:
        #     # At the moment, this part is extremely environment-specific.
        #     # We will refactor this in a future PR.
        #     if isinstance(obs, TinyState):
        #         self._csp_generator = TinyCSPGenerator(
        #             seed=self._seed, explore_method=self._explore_method
        #         )
        #     elif isinstance(obs, PyBulletState):
        #         task_spec = info["task_spec"]
        #         assert isinstance(task_spec, PyBulletTaskSpec)
        #         sim = PyBulletEnv(task_spec, seed=self._seed, use_gui=False)
        #         rom_model = SphericalROMModel(task_spec.human_spec, self._seed)
        #         self._csp_generator = PyBulletCSPGenerator(
        #             sim,
        #             rom_model,
        #             seed=self._seed,
        #             explore_method=self._explore_method,
        #             max_motion_planning_candidates=self._max_motion_planning_candidates,
        #         )
        #     else:
        #         raise NotImplementedError()
        self._sync_csp_generator_train_eval()
        self._recompute_policy(obs)

    def _recompute_policy(self, obs: _ObsType) -> None:
        assert isinstance(self._csp_generator, CSPGenerator)
        csp, samplers, policy, initialization = self._csp_generator.generate(
            obs,
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
        if self._current_policy is None or self._current_policy.check_termination(
            self._last_observation
        ):
            logging.info("Recomputing policy because of termination")
            self._recompute_policy(
                self._last_observation,
            )
        assert self._current_policy is not None
        return self._current_policy.step(self._last_observation)

    def _learn_from_transition(
        self,
        obs: _ObsType,
        act: _ActType,
        next_obs: _ObsType,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        assert np.isclose(reward, 0.0), "Rewards not used in this project!"
        self._csp_generator.learn_from_transition(obs, act, next_obs, done, info)

    def get_step_metrics(self) -> dict[str, float]:
        step_metrics = super().get_step_metrics()
        csp_metrics = self._csp_generator.get_metrics()
        assert not set(csp_metrics) & set(step_metrics), "Metric name conflict"
        step_metrics.update(csp_metrics)
        return step_metrics

    def save(self, model_dir: Path) -> None:
        self._csp_generator.save(model_dir)

    def load(self, model_dir: Path) -> None:
        self._csp_generator.load(model_dir)

    def train(self) -> None:
        super().train()
        self._sync_csp_generator_train_eval()

    def eval(self) -> None:
        super().eval()
        self._sync_csp_generator_train_eval()

    def _sync_csp_generator_train_eval(self) -> None:
        if self._train_or_eval == "train":
            self._csp_generator.train()
        else:
            assert self._train_or_eval == "eval"
            self._csp_generator.eval()
