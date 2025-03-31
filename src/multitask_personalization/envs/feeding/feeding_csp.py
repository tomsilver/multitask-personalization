"""CSP elements for the feeding environment."""

from pathlib import Path
from typing import Any, Collection

from multitask_personalization.csp_generation import CSPGenerator
from multitask_personalization.envs.feeding.feeding_structs import (
    FeedingState,
    FeedingAction,
    MoveToJointPositions,
    CloseGripper,
    MoveToEEPose,
    GraspTool,
)
from multitask_personalization.envs.feeding.feeding_env import FeedingEnv
from multitask_personalization.structs import (
    CSP,
    CSPConstraint,
    CSPCost,
    CSPPolicy,
    CSPSampler,
    CSPVariable,
)


class _FeedingCSPPolicy(CSPPolicy[FeedingState, FeedingAction]):

    def __init__(self, sim: FeedingEnv, csp_variables: Collection[CSPVariable], seed: int = 0) -> None:
        super().__init__(csp_variables=csp_variables, seed=seed)
        self._sim = sim

    def _get_plan(self, obs: FeedingState) -> list[FeedingAction] | None:

        scene_spec = self._sim.scene_spec

        plan: list[FeedingAction] = [
            MoveToJointPositions(scene_spec.retract_pos),
            CloseGripper(),
            MoveToJointPositions(scene_spec.utensil_above_mount_pos),
            MoveToEEPose(scene_spec.utensil_inside_mount),
            GraspTool("utensil"),
            MoveToEEPose(scene_spec.utensil_outside_mount)
        ]

        # self.move_to_joint_positions(self.sim.scene_description.retract_pos)
        # self.close_gripper()
        # self.move_to_joint_positions(self.sim.scene_description.utensil_above_mount_pos)
        # self.move_to_ee_pose(self.sim.scene_description.utensil_inside_mount)
        # self.grasp_tool("utensil")

        # if self.wrist_interface is not None:
        #     time.sleep(1.0) # wait for the utensil to be connected
        #     print("Resetting wrist controller ...")
        #     self.wrist_interface.set_velocity_mode()
        #     self.wrist_interface.reset()

        # self.move_to_ee_pose(self.sim.scene_description.utensil_outside_mount)
        # if self.sim.scene_description.scene_label == "vention":
        #     self.move_to_ee_pose(self.sim.scene_description.utensil_outside_above_mount)
        # elif self.sim.scene_description.scene_label == "wheelchair":
        #     # Not sure if this is necessary.
        #     self.move_to_joint_positions(self.sim.scene_description.retract_pos)
        # # Pre-emptively move to the before_transfer_pos because moving to above_plate_pos from retract_pos is unsafe.
        # self.move_to_joint_positions(self.sim.scene_description.before_transfer_pos)

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
        return action

    def check_termination(self, obs: FeedingState) -> bool:
        return False


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
        return [], {}

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
        return []

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
