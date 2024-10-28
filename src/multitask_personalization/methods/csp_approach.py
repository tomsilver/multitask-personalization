"""An approach that generates and solves a CSP to make decisions."""

from multitask_personalization.methods.approach import BaseApproach, _ActType, _ObsType
from multitask_personalization.structs import CSP, CSPPolicy, CSPSampler
from multitask_personalization.envs.pybullet.pybullet_csp import (
    create_book_handover_csp,
)
from multitask_personalization.utils import solve_csp

import gymnasium as gym


class CSPApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that generates and solves a CSP to make decisions."""

    def __init__(self, action_space: gym.spaces.Space[_ActType], seed: int):
        super().__init__(action_space, seed)
        self._current_policy: CSPPolicy | None = None

    def reset(
        self,
        obs: _ObsType,
    ) -> None:
        super().reset(obs)
        # At the moment, this implementation is extremely specific to the book
        # handover task in the pybullet environment. Will generalize later.
        csp, samplers, policy, initialization = create_book_handover_csp(
            sim, rom_model, preferred_books, self._seed,
        )
        self._current_policy = policy
        sol = solve_csp(csp, initialization, samplers, self._rng)
        self._current_policy.reset(sol)
        import ipdb; ipdb.set_trace()

    def _get_action(self) -> _ActType:
        import ipdb; ipdb.set_trace()

    def _learn_from_transition(
        self,
        obs: _ObsType,
        act: _ActType,
        next_obs: _ObsType,
        reward: float,
        done: bool,
    ) -> None:
        import ipdb; ipdb.set_trace()
