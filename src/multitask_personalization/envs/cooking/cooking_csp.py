"""CSP generation for the cooking environment."""

import abc
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.spaces import Box, Discrete, Tuple
from numpy.typing import NDArray
from tomsutils.spaces import EnumSpace
from tomsutils.utils import create_rng_from_rng

from multitask_personalization.csp_generation import CSPGenerator
from multitask_personalization.envs.cooking.cooking_structs import (
    CookingAction,
    CookingState,
)
from multitask_personalization.structs import (
    CSP,
    CSPConstraint,
    CSPCost,
    CSPPolicy,
    CSPSampler,
    CSPVariable,
    FunctionalCSPConstraint,
    FunctionalCSPSampler,
    LogProbCSPConstraint,
)


class CookingCSPGenerator(CSPGenerator[CookingState, CookingAction]):
    """Generates CSPs for the cooking environment."""

    def save(self, model_dir: Path) -> None:
        # Nothing is learned yet.
        pass

    def load(self, model_dir: Path) -> None:
        # Nothing is learned yet.
        pass

    def _generate_variables(
        self,
        obs: CookingState,
    ) -> tuple[list[CSPVariable], dict[CSPVariable, Any]]:
        import ipdb

        ipdb.set_trace()

    def _generate_personal_constraints(
        self,
        obs: CookingState,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:
        import ipdb

        ipdb.set_trace()

    def _generate_nonpersonal_constraints(
        self,
        obs: CookingState,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:
        import ipdb

        ipdb.set_trace()

    def _generate_exploit_cost(
        self,
        obs: CookingState,
        variables: list[CSPVariable],
    ) -> CSPCost | None:
        return None

    def _generate_samplers(
        self,
        obs: CookingState,
        csp: CSP,
    ) -> list[CSPSampler]:
        import ipdb

        ipdb.set_trace()

    def _generate_policy(
        self,
        obs: CookingState,
        csp: CSP,
    ) -> CSPPolicy:
        import ipdb

        ipdb.set_trace()

    def observe_transition(
        self,
        obs: CookingState,
        act: CookingAction,
        next_obs: CookingState,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        pass
