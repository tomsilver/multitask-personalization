"""An interaction method that selects actions uniformly at random."""

from functools import cached_property

from multitask_personalization.envs.intake_process import (
    IntakeAction,
    IntakeObservation,
)
from multitask_personalization.interaction.interaction_method import InteractionMethod


class RandomInteractionMethod(InteractionMethod):
    """An interaction method that selects actions uniformly at random."""

    @cached_property
    def _ordered_actions(self) -> list[IntakeAction]:
        return sorted(self._action_space)

    def get_action(self) -> IntakeAction:
        idx = self._rng.choice(len(self._ordered_actions))
        return self._ordered_actions[idx]

    def observe(self, obs: IntakeObservation) -> None:
        pass
