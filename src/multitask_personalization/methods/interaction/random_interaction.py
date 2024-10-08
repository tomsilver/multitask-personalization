"""An interaction method that selects actions uniformly at random."""

from multitask_personalization.envs.intake_process import (
    IntakeAction,
    IntakeObservation,
)
from multitask_personalization.methods.interaction.interaction_method import (
    InteractionMethod,
)


class RandomInteractionMethod(InteractionMethod):
    """An interaction method that selects actions uniformly at random."""

    def get_action(self) -> IntakeAction:
        assert self._current_action_space is not None
        return self._current_action_space.sample()

    def observe(self, obs: IntakeObservation) -> None:
        pass
