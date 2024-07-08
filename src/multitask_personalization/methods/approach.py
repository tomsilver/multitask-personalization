"""An approach that combines calibration, interaction, and policies."""

from multitask_personalization.envs.intake_process import (
    IntakeAction,
    IntakeObservation,
)
from multitask_personalization.envs.mdp import MDPAction, MDPState
from multitask_personalization.methods.calibration.calibrator import Calibrator
from multitask_personalization.methods.interaction.interaction_method import (
    InteractionMethod,
)
from multitask_personalization.methods.policies.parameterized_policy import (
    ParameterizedPolicy,
)


class Approach:
    """An approach that combines calibration, interaction, and policies."""

    def __init__(
        self,
        calibrator: Calibrator,
        interaction_method: InteractionMethod,
        policy: ParameterizedPolicy,
    ) -> None:
        self._calibrator = calibrator
        self._interaction_method = interaction_method
        self._policy = policy
        # History of (task ID, [(action, observation)]).
        self._intake_history: list[
            tuple[str, list[tuple[IntakeAction, IntakeObservation]]]
        ] = []
        self._last_intake_action: IntakeAction | None = None

    def reset(self, task_id: str) -> None:
        """Called when a new task begins."""
        self._intake_history.append((task_id, []))

    def get_intake_action(self) -> None:
        """Get a next intake action to execute."""
        action = self._interaction_method.get_action()
        self._last_intake_action = action
        return action

    def record_intake_observation(self, obs: IntakeObservation) -> None:
        """Record the response to an intake action."""
        assert self._last_intake_action is not None
        self._intake_history[-1][1].append((self._last_intake_action, obs))
        self._interaction_method.observe(obs)

    def finish_intake(self) -> None:
        """Calibrate the policy for this task."""
        task_id, data = self._intake_history[-1]
        parameters = self._calibrator.get_parameters(task_id, data)
        self._policy.reset(task_id, parameters)

    def get_mdp_action(self, state: MDPState) -> MDPAction:
        """Get an action for the MDP."""
        return self._policy.step(state)
