"""A domain-specific parameter setting method for grid world tasks."""

from typing import Collection

from multitask_personalization.envs.grid_world.grid_world_intake_process import (
    _RewardValueQuestion,
)
from multitask_personalization.envs.grid_world.grid_world_mdp import _GridState
from multitask_personalization.envs.grid_world.grid_world_tasks import (
    GridTask,
)
from multitask_personalization.envs.intake_process import (
    IntakeAction,
    IntakeObservation,
)
from multitask_personalization.methods.calibration.calibrator import Calibrator
from multitask_personalization.methods.policies.parameterized_policy import (
    PolicyParameters,
)
from multitask_personalization.utils import topological_sort


class GridWorldCalibrator(Calibrator):
    """A domain-specific parameter setting method for grid world tasks."""

    def __init__(self, terminal_locs: Collection[tuple[int, int]]) -> None:
        self._terminal_locs = terminal_locs

    def get_parameters(
        self, task_id: str, intake_data: list[tuple[IntakeAction, IntakeObservation]]
    ) -> PolicyParameters:
        # For now, just use the reward value actions to order the terminal locs
        # and find the maximal one. Then use that as the policy parameters.
        pairwise_relations: list[tuple[_GridState, _GridState]] = []
        for action, obs in intake_data:
            if isinstance(action, _RewardValueQuestion):
                loc1, loc2 = action.loc1, action.loc2
                if obs:
                    pairwise_relations.append((loc1, loc2))
        terminal_locs = sorted(self._terminal_locs)
        ordered_locs = topological_sort(terminal_locs, pairwise_relations)
        return ordered_locs[-1]


class OracleGridWorldCalibrator(Calibrator):
    """A domain-specific calibrator for grid world that uses oracle info."""

    def __init__(self, tasks: Collection[GridTask]) -> None:
        self._task_id_to_params: dict[str, tuple[int, int]] = {}
        for task in tasks:
            terminal_rewards = (
                task._terminal_rewards
            )  # pylint: disable=protected-access
            _, param = max((r, l) for l, r in terminal_rewards.items())
            self._task_id_to_params[task.id] = param

    def get_parameters(
        self, task_id: str, intake_data: list[tuple[IntakeAction, IntakeObservation]]
    ) -> PolicyParameters:
        return self._task_id_to_params[task_id]
