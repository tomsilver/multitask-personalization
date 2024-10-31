"""An approach that generates and solves a CSP to make decisions."""

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
)
from multitask_personalization.utils import solve_csp


class CSPApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that generates and solves a CSP to make decisions."""

    def __init__(
        self,
        action_space: gym.spaces.Space[_ActType],
        seed: int,
        explore_epsilon: float = 0.1,
    ):
        super().__init__(action_space, seed)
        self._explore_epsilon = explore_epsilon
        self._current_policy: CSPPolicy | None = None
        self._csp_generator: CSPGenerator | None = None

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
                self._csp_generator = TinyCSPGenerator(self._seed)
            elif isinstance(obs, PyBulletState):
                task_spec = info["task_spec"]
                assert isinstance(task_spec, PyBulletTaskSpec)
                sim = PyBulletEnv(task_spec, seed=self._seed, use_gui=True)
                rom_model = SphericalROMModel(task_spec.human_spec, self._seed)
                preferred_books = ["book2"]  # coming soon: learning this
                self._csp_generator = PyBulletCSPGenerator(
                    sim, rom_model, preferred_books, self._seed
                )
            else:
                raise NotImplementedError()
        explore = self._rng.uniform() < self._explore_epsilon
        assert isinstance(self._csp_generator, CSPGenerator)
        csp, samplers, policy, initialization = self._csp_generator.generate(
            obs, explore=explore
        )
        sol = solve_csp(
            csp,
            initialization,
            samplers,
            self._rng,
        )
        self._current_policy = policy
        self._current_policy.reset(sol)

    def _get_action(self) -> _ActType:
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
