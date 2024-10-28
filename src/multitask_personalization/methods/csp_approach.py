"""An approach that generates and solves a CSP to make decisions."""

from typing import Any

import gymnasium as gym

from multitask_personalization.envs.pybullet.pybullet_csp import (
    create_book_handover_csp,
)
from multitask_personalization.envs.pybullet.pybullet_env import PyBulletEnv
from multitask_personalization.envs.pybullet.pybullet_task_spec import PyBulletTaskSpec
from multitask_personalization.methods.approach import BaseApproach, _ActType, _ObsType
from multitask_personalization.structs import (
    CSP,
    CSPPolicy,
    CSPSampler,
    CSPVariable,
    TrainableCSPConstraint,
)
from multitask_personalization.utils import solve_csp


class CSPApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that generates and solves a CSP to make decisions."""

    def __init__(self, action_space: gym.spaces.Space[_ActType], seed: int):
        super().__init__(action_space, seed)
        self._current_csp: CSP | None = None
        self._current_samplers: list[CSPSampler] = []
        self._current_csp_initialization: dict[CSPVariable, Any] = {}
        self._current_policy: CSPPolicy | None = None

    def reset(
        self,
        obs: _ObsType,
        info: dict[str, Any],
    ) -> None:
        super().reset(obs)
        # At the moment, this implementation is extremely specific to the book
        # handover task in the pybullet environment. Will generalize later.
        # For efficiency, and because the task spec is not changing right now,
        # we only create the CSP one time.
        if self._current_csp is None:
            task_spec = info["task_spec"]
            assert isinstance(task_spec, PyBulletTaskSpec)
            sim = PyBulletEnv(task_spec, seed=self._seed)
            csp, samplers, policy, initialization = create_book_handover_csp(
                sim,
                self._seed,
            )
            self._current_csp = csp
            self._current_samplers = samplers
            self._current_policy = policy
            self._current_csp_initialization = initialization
        # Re-solve the CSP because the constraints may have changed internally
        # as a result of training.
        sol = solve_csp(
            self._current_csp,
            self._current_csp_initialization,
            self._current_samplers,
            self._rng,
        )
        self._current_policy.reset(sol)
        # Reset learnable constraints.
        for constraint in self._get_trainable_constraints():
            constraint.reset(obs)

    def _get_action(self) -> _ActType:
        return self._current_policy.step(self._last_observation)

    def _learn_from_transition(
        self,
        obs: _ObsType,
        act: _ActType,
        next_obs: _ObsType,
        reward: float,
        done: bool,
    ) -> None:
        # Update the trainable constraints. Right now this is done serially but
        # it could be parallelized in the future.
        for constraint in self._get_trainable_constraints():
            constraint.update(act, next_obs)

    def _get_trainable_constraints(self) -> list[TrainableCSPConstraint]:
        assert self._current_csp is not None
        return [
            c
            for c in self._current_csp.constraints
            if isinstance(c, TrainableCSPConstraint)
        ]
