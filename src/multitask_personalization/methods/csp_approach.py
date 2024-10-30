"""An approach that generates and solves a CSP to make decisions."""

from typing import Any

import gymnasium as gym

from multitask_personalization.envs.tiny.tiny_csp import TinyCSPGenerator
from multitask_personalization.envs.tiny.tiny_env import TinyState
from multitask_personalization.methods.approach import BaseApproach, _ActType, _ObsType
from multitask_personalization.structs import (
    CSPGenerator,
    CSPPolicy,
)
from multitask_personalization.utils import solve_csp


class CSPApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that generates and solves a CSP to make decisions."""

    def __init__(self, action_space: gym.spaces.Space[_ActType], seed: int):
        super().__init__(action_space, seed)
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
            else:
                raise NotImplementedError()
        csp, samplers, policy, initialization = self._csp_generator.generate(obs)
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
