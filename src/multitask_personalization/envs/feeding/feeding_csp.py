"""CSP elements for the feeding environment."""

from pathlib import Path
from typing import Any, Collection

import numpy as np
from gymnasium.spaces import Box
from pybullet_helpers.geometry import Pose

from multitask_personalization.csp_generation import CSPGenerator
from multitask_personalization.envs.feeding.feeding_env import FeedingEnv
from multitask_personalization.envs.feeding.feeding_structs import (
    CloseGripper,
    FeedingAction,
    FeedingState,
    GraspTool,
    MovePlate,
    MoveToEEPose,
    MoveToJointPositions,
    UngraspTool,
    WaitForUserInput,
)
from multitask_personalization.structs import (
    CSP,
    CSPConstraint,
    CSPCost,
    CSPPolicy,
    CSPSampler,
    CSPVariable,
    FunctionalCSPSampler,
)


class _FeedingCSPPolicy(CSPPolicy[FeedingState, FeedingAction]):

    def __init__(
        self, sim: FeedingEnv, csp_variables: Collection[CSPVariable], seed: int = 0
    ) -> None:
        super().__init__(csp_variables=csp_variables, seed=seed)
        self._sim = sim
        self._current_plan: list[FeedingAction] = []
        self._terminated = False

    def _get_plan(self, obs: FeedingState) -> list[FeedingAction] | None:

        scene_spec = self._sim.scene_spec

        current_plate_pose = obs.plate_pose
        new_plate_x, new_plate_y = self._get_value("plate_position")
        new_plate_pose = Pose(
            (
                new_plate_x,
                new_plate_y,
                current_plate_pose.position[2],
            ),
            current_plate_pose.orientation,
        )

        move_plate_plan: list[FeedingAction] = [MovePlate(new_plate_pose)]

        pick_utensil_plan: list[FeedingAction] = [
            MoveToJointPositions(scene_spec.retract_pos),
            CloseGripper(),
            MoveToJointPositions(scene_spec.utensil_above_mount_pos),
            MoveToEEPose(scene_spec.utensil_inside_mount),
            GraspTool("utensil"),
            MoveToEEPose(scene_spec.utensil_outside_mount),
            MoveToEEPose(scene_spec.utensil_outside_above_mount),
            MoveToJointPositions(scene_spec.before_transfer_pos),
        ]

        acquire_bite_plan: list[FeedingAction] = [
            MoveToJointPositions(scene_spec.above_plate_pos),
        ]

        transfer_bite_plan: list[FeedingAction] = [
            MoveToJointPositions(scene_spec.before_transfer_pos),
            MoveToEEPose(scene_spec.before_transfer_pose),
            MoveToEEPose(scene_spec.outside_mouth_transfer_pose),
            MoveToEEPose(scene_spec.before_transfer_pose),
        ]

        stow_utensil_plan: list[FeedingAction] = [
            MoveToJointPositions(scene_spec.utensil_outside_above_mount_pos),
            MoveToEEPose(scene_spec.utensil_outside_mount),
            MoveToEEPose(scene_spec.utensil_inside_mount),
            UngraspTool(),
            MoveToEEPose(scene_spec.utensil_above_mount),
            MoveToJointPositions(scene_spec.retract_pos),
        ]

        finish = [WaitForUserInput("done")]

        plan = (
            move_plate_plan
            + pick_utensil_plan
            + acquire_bite_plan
            + transfer_bite_plan
            + stow_utensil_plan
            + finish
        )

        return plan

    def reset(self, solution: dict[CSPVariable, Any]) -> None:
        super().reset(solution)
        self._current_plan = []
        self._terminated = False

    def step(self, obs: FeedingState) -> FeedingAction:
        if not self._current_plan:
            self._sim.set_state(obs)
            plan = self._get_plan(obs)
            assert plan is not None
            self._current_plan = plan
        action = self._current_plan.pop(0)
        self._terminated = (
            isinstance(action, WaitForUserInput) and action.user_input == "done"
        )
        return action

    def check_termination(self, obs: FeedingState) -> bool:
        return self._terminated


class FeedingCSPGenerator(CSPGenerator[FeedingState, FeedingAction]):
    """Generate CSPs for the feeding environment."""

    def __init__(self, sim: FeedingEnv, *args, **kwargs) -> None:
        self._sim = sim
        super().__init__(*args, **kwargs)

    def save(self, model_dir: Path) -> None:
        print("WARNING: saving not yet implemented for FeedingCSPGenerator.")

    def load(self, model_dir: Path) -> None:
        print("WARNING: loading not yet implemented for FeedingCSPGenerator.")

    def _generate_variables(
        self,
        obs: FeedingState,
    ) -> tuple[list[CSPVariable], dict[CSPVariable, Any]]:

        # XY position of the plate.
        plate_position_domain = Box(
            np.array(self._sim.scene_spec.plate_position_lower),
            np.array(self._sim.scene_spec.plate_position_upper),
            dtype=np.float32,
        )
        plate_position = CSPVariable("plate_position", plate_position_domain)
        init_plate_position = (obs.plate_pose.position[0], obs.plate_pose.position[1])

        return [plate_position], {
            plate_position: init_plate_position,
        }

    def _generate_personal_constraints(
        self,
        obs: FeedingState,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:
        return []

    def _generate_nonpersonal_constraints(
        self,
        obs: FeedingState,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:
        return []

    def _generate_exploit_cost(
        self,
        obs: FeedingState,
        variables: list[CSPVariable],
    ) -> CSPCost | None:
        return None

    def _generate_samplers(
        self,
        obs: FeedingState,
        csp: CSP,
    ) -> list[CSPSampler]:

        # Sample plate positions.
        plate_position = csp.variables[0]

        def _sample_plate_position(
            _: dict[CSPVariable, Any], rng: np.random.Generator
        ) -> dict[CSPVariable, Any]:
            new_pos = rng.uniform(
                low=self._sim.scene_spec.plate_position_lower,
                high=self._sim.scene_spec.plate_position_upper,
            ).astype(np.float32)
            return {plate_position: new_pos}

        plate_position_sampler = FunctionalCSPSampler(
            _sample_plate_position, csp, {plate_position}
        )

        return [plate_position_sampler]

    def _generate_policy(
        self,
        obs: FeedingState,
        csp_variables: Collection[CSPVariable],
    ) -> CSPPolicy:
        return _FeedingCSPPolicy(self._sim, csp_variables, self._seed)

    def observe_transition(
        self,
        obs: FeedingState,
        act: FeedingAction,
        next_obs: FeedingState,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        pass
