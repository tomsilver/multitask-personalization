"""An approach that combines calibration, interaction, and policies."""

from typing import Generic, TypeVar

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
    PolicyParameters,
)

_U = TypeVar("_U", bound=IntakeAction)
_O = TypeVar("_O", bound=IntakeObservation)
_S = TypeVar("_S", bound=MDPState)
_A = TypeVar("_A", bound=MDPAction)
_P = TypeVar("_P", bound=PolicyParameters)


class Approach(Generic[_U, _O, _P, _S, _A]):
    """An approach that combines calibration, interaction, and policies."""

    def __init__(
        self,
        calibrator: Calibrator[_U, _O, _P],
        interaction_method: InteractionMethod[_U, _O],
        policy: ParameterizedPolicy[_S, _A, _P],
    ) -> None:
        self._calibrator = calibrator
        self._interaction_method = interaction_method
        self._policy = policy
        # History of (task ID, [(action, observation)]).
        self._intake_history: list[tuple[str, list[tuple[_U, _O]]]] = []
        self._last_intake_action: _U | None = None

    def reset(
        self,
        task_id: str,
        action_space: set[_U],
        observation_space: set[_O],
    ) -> None:
        """Called when a new task begins."""
        self._intake_history.append((task_id, []))
        self._interaction_method.reset(task_id, action_space, observation_space)

    def get_intake_action(self) -> _U:
        """Get a next intake action to execute."""
        action = self._interaction_method.get_action()
        self._last_intake_action = action
        return action

    def record_intake_observation(self, obs: _O) -> None:
        """Record the response to an intake action."""
        assert self._last_intake_action is not None
        self._intake_history[-1][1].append((self._last_intake_action, obs))
        self._interaction_method.observe(obs)

    def finish_intake(self) -> None:
        """Calibrate the policy for this task."""
        task_id, data = self._intake_history[-1]
        parameters = self._calibrator.get_parameters(task_id, data)
        self._policy.reset(task_id, parameters)

    def get_mdp_action(self, state: _S) -> _A:
        """Get an action for the MDP."""
        return self._policy.step(state)
