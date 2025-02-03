"""An approach that generates and solves a CSP to make decisions."""

import logging
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from tomsutils.llm import LargeLanguageModel

from multitask_personalization.csp_generation import CSPGenerator
from multitask_personalization.csp_solvers import CSPSolver
from multitask_personalization.envs.cooking.cooking_csp import CookingCSPGenerator
from multitask_personalization.envs.cooking.cooking_hidden_spec import (
    MealSpecMealPreferenceModel,
)
from multitask_personalization.envs.cooking.cooking_scene_spec import CookingSceneSpec
from multitask_personalization.envs.pybullet.pybullet_csp import PyBulletCSPGenerator
from multitask_personalization.envs.pybullet.pybullet_env import PyBulletEnv
from multitask_personalization.envs.pybullet.pybullet_scene_spec import (
    PyBulletSceneSpec,
)
from multitask_personalization.envs.tiny.tiny_csp import TinyCSPGenerator
from multitask_personalization.envs.tiny.tiny_env import TinySceneSpec
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
    PublicSceneSpec,
)
from multitask_personalization.utils import visualize_csp_graph


class CSPApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that generates and solves a CSP to make decisions."""

    def __init__(
        self,
        scene_spec: PublicSceneSpec,
        action_space: gym.spaces.Space[_ActType],
        csp_solver: CSPSolver,
        llm: LargeLanguageModel | None = None,
        max_motion_planning_candidates: int = 1,
        explore_method: str = "nothing-personal",
        disable_learning: bool = False,
        csp_save_dir: str | None = None,
        seed: int = 0,
    ):
        super().__init__(scene_spec, action_space, seed)
        self._llm = llm
        self._csp_solver = csp_solver
        self._current_policy: CSPPolicy | None = None
        self._current_sol: dict[CSPVariable, Any] | None = None
        self._explore_method = explore_method
        self._disable_learning = disable_learning
        self._max_motion_planning_candidates = max_motion_planning_candidates
        self._csp_save_dir = Path(csp_save_dir) if csp_save_dir else None
        self._csp_generator = self._create_csp_generator()

    def reset(
        self,
        obs: _ObsType,
        info: dict[str, Any],
    ) -> None:
        super().reset(obs, info)
        self._sync_csp_generator_train_eval()
        self._recompute_policy(obs)

    def _recompute_policy(self, obs: _ObsType) -> None:
        assert isinstance(self._csp_generator, CSPGenerator)
        csp, samplers, policy, initialization = self._csp_generator.generate(
            obs,
        )
        # Save the generated CSP.
        if self._csp_save_dir is not None:
            self._csp_save_dir.mkdir(exist_ok=True)
            while True:
                time_str = time.strftime("%Y%m%d-%H%M%S")
                viz_file = self._csp_save_dir / f"csp_{time_str}.png"
                if viz_file.exists():
                    time.sleep(1)
                else:
                    break
            visualize_csp_graph(csp, viz_file)
        self._current_sol = self._csp_solver.solve(csp, initialization, samplers)
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
        self._csp_generator.observe_transition(obs, act, next_obs, done, info)

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

    def _create_csp_generator(self) -> CSPGenerator:
        if isinstance(self._scene_spec, TinySceneSpec):
            return TinyCSPGenerator(
                seed=self._seed,
                explore_method=self._explore_method,
                disable_learning=self._disable_learning,
            )
        if isinstance(self._scene_spec, PyBulletSceneSpec):
            assert self._llm is not None
            sim = PyBulletEnv(
                self._scene_spec, self._llm, seed=self._seed, use_gui=False
            )
            rom_model = SphericalROMModel(self._scene_spec.human_spec, self._seed)
            return PyBulletCSPGenerator(
                sim,
                rom_model,
                self._llm,
                seed=self._seed,
                explore_method=self._explore_method,
                disable_learning=self._disable_learning,
                max_motion_planning_candidates=self._max_motion_planning_candidates,
            )
        if isinstance(self._scene_spec, CookingSceneSpec):
            meal_model = MealSpecMealPreferenceModel(
                self._scene_spec.universal_meal_specs
            )
            return CookingCSPGenerator(
                self._scene_spec,
                meal_model,
                explore_method=self._explore_method,
                disable_learning=self._disable_learning,
            )
        raise NotImplementedError()

    def _sync_csp_generator_train_eval(self) -> None:
        if self._train_or_eval == "train":
            self._csp_generator.train()
        else:
            assert self._train_or_eval == "eval"
            self._csp_generator.eval()
