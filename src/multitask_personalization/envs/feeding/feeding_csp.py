"""CSP elements for the feeding environment."""

import logging
from pathlib import Path
from typing import Any, Callable, Collection
from functools import partial

import numpy as np
from gymnasium.spaces import Box, Discrete
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, set_pose
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    check_body_collisions,
    inverse_kinematics,
    set_robot_joints_with_held_object,
)
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from tomsutils.llm import LargeLanguageModel
from tomsutils.spaces import EnumSpace

from multitask_personalization.csp_generation import CSPGenerator
from multitask_personalization.envs.feeding.feeding_env import FeedingEnv
from multitask_personalization.envs.feeding.feeding_scene_spec import FeedingSceneSpec
from multitask_personalization.envs.feeding.feeding_structs import (
    FeedingAction,
    FeedingObservation,
    FeedingInitializationQueryObservation,
    FeedingInitializationDatasetObservation,
    FeedingOcclusionQueryObservation,
    FeedingOcclusionDatasetObservation,
    FeedingObservationWithContext,
    FeedingInitializationAction,
    FeedingPlateDrinkAction,
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
)
from multitask_personalization.utils import Threshold1DModel


class _FeedingCSPPolicy(CSPPolicy[FeedingObservation, FeedingAction]):

    def __init__(
        self, sim: FeedingEnv, csp_variables: Collection[CSPVariable], seed: int = 0
    ) -> None:
        super().__init__(csp_variables=csp_variables, seed=seed)
        self._sim = sim

    def step(self, obs: FeedingObservation) -> FeedingAction:
        if isinstance(obs, FeedingInitializationQueryObservation):
            feeding_side = self._get_value("feeding_side")
            bite_ordering = obs.bite_ordering_options[self._get_value("bite_ordering")]
            ready_signal = self._get_value("ready_signal")
            be_verbal = self._get_value("be_verbal")
            return FeedingInitializationAction(
                feeding_side=feeding_side,
                bite_ordering=bite_ordering,
                ready_signal=ready_signal,
                be_verbal=be_verbal,
            )
        if isinstance(obs, FeedingOcclusionQueryObservation):
            planned_plate_position = self._get_value("plate_position")
            plate_delta_xy = (obs.plate_pose.position[0] - planned_plate_position[0],
                              obs.plate_pose.position[1] - planned_plate_position[1])
            planned_plate_pose = _plate_position_to_pose(planned_plate_position)
            before_transfer_pose = _transform_pose_relative_to_plate(
                "before_transfer_pose", planned_plate_pose, self._sim.scene_spec
            )
            before_transfer_pos = _transform_joints_relative_to_plate(
                "before_transfer_pos", planned_plate_pose, self._sim.robot, self._sim.scene_spec
            )
            above_plate_pos = _transform_joints_relative_to_plate(
                "above_plate_pos", planned_plate_pose, self._sim.robot, self._sim.scene_spec,
            )
            planned_drink_position = self._get_value("drink_position")
            drink_delta_xy = (obs.drink_pose.position[0] - planned_drink_position[0],
                              obs.drink_pose.position[1] - planned_drink_position[1])
            return FeedingPlateDrinkAction(plate_delta_xy=plate_delta_xy,
                                           drink_delta_xy=drink_delta_xy,
                                           before_transfer_pose=before_transfer_pose,
                                           before_transfer_pos=before_transfer_pos,
                                           above_plate_pos=above_plate_pos)
        raise NotImplementedError

    def check_termination(self, obs: FeedingObservation) -> bool:
        return False
    

class LLMMultipleChoiceConstraintModel:
    """Shared code for table side, bite ordering, etc."""

    def __init__(
        self,
        name: str,
        llm: LargeLanguageModel,
        get_choices_from_observation: Callable[[FeedingObservation], list[Any]],
        get_variable_value_from_choice: Callable[[Any, FeedingObservation], Any] | None = None,
        seed: int = 0,
    ):
        self.name = name  # this must match the name in FeedingInitializationDatasetObservation
        self.llm = llm
        self.get_choices_from_observation = get_choices_from_observation
        self.get_variable_value_from_choice = get_variable_value_from_choice or (lambda x, o: x)
        self.summary_preferences = "Unknown"
        self.seed = seed
        self.initialization_data_obs_history: list[FeedingInitializationDatasetObservation] = []

    def create_constraint(self, obs: FeedingObservation, variable: CSPVariable) -> CSPConstraint:
        assert variable.name == self.name
        preference_constraint = FunctionalCSPConstraint(
            f"{self.name}_preference",
            [variable],
            partial(self.choice_most_preferred, obs),
        )
        return preference_constraint
    
    def create_sampler(self, obs: FeedingObservation, variable: CSPVariable, csp: CSP) -> CSPSampler:
        assert variable.name == self.name

        def _sample_fn(_sol: dict[CSPVariable, Any], _rng: np.random.Generator) -> Any:
            return {variable: self.get_most_preferred_choice(obs)}

        sampler = FunctionalCSPSampler(
            _sample_fn,
            csp,
            {variable},
        )

        return sampler
    
    def get_most_preferred_choice(self, obs: FeedingObservation) -> Any:
        choices = self.get_choices_from_observation(obs)
        choice_nums = [str(i+1) for i in range(len(choices))]
        prompt = self.get_prompt(obs, choices)
        logprobs, _ = self.llm.get_multiple_choice_logprobs(prompt, choice_nums, self.seed)
        selected_num_str = max(choice_nums, key=logprobs.get)
        assert selected_num_str.isdigit()
        choice_idx = int(selected_num_str) - 1
        choice = choices[choice_idx]
        return self.get_variable_value_from_choice(choice, obs)

    def choice_most_preferred(self, obs: FeedingObservation, choice: Any) -> bool:
        most_preferred_choice = self.get_most_preferred_choice(obs)
        return most_preferred_choice == choice
    
    def get_prompt(self, obs: FeedingObservation, choices: list[Any]) -> str:
        choice_list_str = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
        choice_example_list = " or ".join([f"'{i+1}'" for i in range(len(choices))])
        context_str = self.get_context_str(obs)
        prompt = f"""You are a mealtime assistance robot and you are choosing a value for a variable `{self.name}` which can take on the following values:

{choice_list_str}

The current context is:

{context_str}

Based on your past interactions with the user, you have the following estimation of their preferences:

{self.summary_preferences}

Which choice should you make? Return only the number of the choice, e.g., {choice_example_list}.
"""

        return prompt
    
    def get_context_str(self, obs: FeedingObservation) -> str:
        assert isinstance(obs, FeedingObservationWithContext)
        return obs.get_context_str() 
        
    def learn_incremental(self, obs: FeedingInitializationDatasetObservation) -> None:
        self.initialization_data_obs_history.append(obs)
        history_str = self.get_history_str()
        prompt = f"""You are a mealtime assistance robot and you are summarizing a user's preferences for a variable `{self.name}`. The preferences depend on context. Here is a history of (context, user choice):

{history_str}

Based on this history, summarize the user's contextual preferences.
"""

        response, _ = self.llm.query(prompt, temperature=1.0, seed=self.seed)
        self.summary_preferences = response
        print(f"Updated user preferences for {self.name}: {self.summary_preferences}")
    
    def get_history_str(self) -> str:
        combined_str = ""
        for obs in self.initialization_data_obs_history:
            user_choice = getattr(obs, self.name)
            combined_str += f"\nCONTEXT: {obs.get_context_str()}"
            combined_str += f"\nUSER CHOICE: {user_choice}\n"
        return combined_str
    
    

class FeedingCSPGenerator(CSPGenerator[FeedingObservation, FeedingAction]):
    """Generate CSPs for the feeding environment."""

    def __init__(
        self, sim: FeedingEnv, llm: LargeLanguageModel, occlusion_scale_model: Threshold1DModel, *args, **kwargs
    ) -> None:
        self._sim = sim
        self._llm = llm
        self._occlusion_model = occlusion_scale_model
        super().__init__(*args, **kwargs)
        self._feeding_side_model = LLMMultipleChoiceConstraintModel("feeding_side", llm, lambda o: ["left", "right"])
        self._bite_ordering_model = LLMMultipleChoiceConstraintModel("bite_ordering", llm,lambda o: o.bite_ordering_options,
                                                                     get_variable_value_from_choice=lambda x,o: o.bite_ordering_options.index(x))
        self._ready_signal_model = LLMMultipleChoiceConstraintModel("ready_signal", llm, lambda o: ["mouth_open", "button", "auto_continue"])
        self._be_verbal_model = LLMMultipleChoiceConstraintModel("be_verbal", llm, lambda o: [True, False])

    def save(self, model_dir: Path) -> None:
        print("WARNING: saving not yet implemented for FeedingCSPGenerator.")

    def load(self, model_dir: Path) -> None:
        print("WARNING: loading not yet implemented for FeedingCSPGenerator.")

    def _generate_variables(
        self,
        obs: FeedingObservation,
    ) -> tuple[list[CSPVariable], dict[CSPVariable, Any]]:
        
        if isinstance(obs, FeedingInitializationQueryObservation):
            feeding_side = CSPVariable("feeding_side", EnumSpace(["left", "right"]))
            bite_ordering = CSPVariable("bite_ordering", Discrete(len(obs.bite_ordering_options)))  # index into obs.bite_ordering_options
            ready_signal = CSPVariable("ready_signal", EnumSpace(["mouth_open", "button", "auto_continue"]))
            be_verbal = CSPVariable("be_verbal", EnumSpace([True, False]))
            variables = [feeding_side, bite_ordering, ready_signal, be_verbal]

            initialization = {
                feeding_side: "left",
                bite_ordering: 0,
                ready_signal: "mouth_open",
                be_verbal: True,
            }

            return variables, initialization
        
        if isinstance(obs, FeedingInitializationDatasetObservation):
            return [], {}
        
        if isinstance(obs, FeedingOcclusionQueryObservation):
            # Plate position variable.
            plate_position_domain = Box(
                np.array([-np.inf, -np.inf]),
                np.array([np.inf, np.inf]),
                dtype=np.float32,
            )
            plate_position = CSPVariable("plate_position", plate_position_domain)
            init_plate_position = (obs.plate_pose.position[0], obs.plate_pose.position[1])

            # Drink position variable.
            drink_position_domain = Box(
                np.array([-np.inf, -np.inf]),
                np.array([np.inf, np.inf]),
                dtype=np.float32,
            )
            drink_position = CSPVariable("drink_position", drink_position_domain)
            init_drink_position = (obs.drink_pose.position[0], obs.drink_pose.position[1])

            variables = [plate_position, drink_position]
            initialization = {
                plate_position: init_plate_position,
                drink_position: init_drink_position,
            }

            return variables, initialization
        
        if isinstance(obs, FeedingOcclusionDatasetObservation):
            return [], {}

        raise NotImplementedError

    def _generate_personal_constraints(
        self,
        obs: FeedingObservation,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:

        constraints: list[CSPConstraint] = []
        
        # Add LLM constraints.
        if isinstance(obs, FeedingInitializationQueryObservation):
            for model, variable in zip([self._feeding_side_model, self._bite_ordering_model, self._ready_signal_model, self._be_verbal_model], variables, strict=True):
                assert model.name == variable.name  # sanity check
                constraint = model.create_constraint(obs, variable)
                constraints.append(constraint)

        # Add occlusion scale constraint.
        if isinstance(obs, FeedingOcclusionQueryObservation):
            plate_position, drink_position = variables

            # NOTE: we are currently just using the MLE occlusion scale, rather than
            # using the full distribution. That means that "ours" will be equivalent
            # to "exploit_only". This is because we're not really running full
            # experiments in this environment.
            occlusion_scale = (
                1.0 - (self._occlusion_model.post_max + self._occlusion_model.post_min) / 2
            )
            # occlusion_scale = 0.999
            self._sim.set_occlusion_scale(occlusion_scale)
            logging.info(f"Set sim occlusion scale to {occlusion_scale:.3f}")
            
            def _user_view_unoccluded_by_utensil(
                plate_position: NDArray[np.float32],
            ) -> bool:
                self._sim.sync_from_observation(obs)
                new_plate_pose = _plate_position_to_pose(plate_position, obs.plate_pose)
                field_name = "above_plate_pos"
                try:
                    robot_joints = _transform_joints_relative_to_plate(
                        field_name,
                        new_plate_pose,
                        self._sim.robot,
                        self._sim.scene_spec,
                        arm_joints_only=False,
                    )
                except InverseKinematicsError:
                    print("WARNING: IK failed within _user_view_unoccluded_by_utensil()")
                    # from pybullet_helpers.gui import visualize_pose
                    # visualize_pose(new_plate_pose, self._sim.physics_client_id)
                    return False
                held_object_id = self._sim.get_object_id_from_name("utensil")
                held_object_tf = self._sim.scene_spec.utensil_held_object_tf
                set_robot_joints_with_held_object(
                    self._sim.robot,
                    self._sim.physics_client_id,
                    held_object_id,
                    held_object_tf,
                    robot_joints,
                )
                self._sim.robot.set_finger_state(
                    self._sim.scene_spec.tool_grasp_fingers_value
                )
                return not self._sim.robot_in_occlusion()

            user_view_unoccluded_by_utensil_constraint = FunctionalCSPConstraint(
                "user_view_unoccluded_by_utensil",
                [plate_position],
                _user_view_unoccluded_by_utensil,
            )

            constraints.append(user_view_unoccluded_by_utensil_constraint)

            def _user_view_unoccluded_by_drink(
                drink_position: NDArray[np.float32],
            ) -> bool:
                self._sim.sync_from_observation(obs)
                new_drink_pose = _drink_position_to_pose(drink_position, obs.drink_pose)
                drink_post_grasp_pose = _transform_pose_relative_to_drink(
                    "drink_default_post_grasp_pose", new_drink_pose, self._sim.scene_spec
                )
                # from pybullet_helpers.gui import visualize_pose
                # visualize_pose(new_drink_pose, self._sim.physics_client_id)
                try:
                    robot_joints = inverse_kinematics(
                        self._sim.robot, drink_post_grasp_pose
                    )
                except InverseKinematicsError:
                    print("WARNING: IK failed within _user_view_unoccluded_by_drink()")
                    return False
                held_object_id = self._sim.get_object_id_from_name("drink")
                held_object_tf = self._sim.scene_spec.drink_held_object_tf
                set_robot_joints_with_held_object(
                    self._sim.robot,
                    self._sim.physics_client_id,
                    held_object_id,
                    held_object_tf,
                    robot_joints,
                )
                self._sim.robot.set_finger_state(
                    self._sim.scene_spec.tool_grasp_fingers_value
                )
                return not self._sim.robot_in_occlusion()

            user_view_unoccluded_by_drink_constraint = FunctionalCSPConstraint(
                "user_view_unoccluded_by_drink",
                [drink_position],
                _user_view_unoccluded_by_drink,
            )
            constraints.append(user_view_unoccluded_by_drink_constraint)

        return constraints

    def _generate_nonpersonal_constraints(
        self,
        obs: FeedingObservation,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:

        constraints: list[CSPConstraint] = []

        if isinstance(obs, FeedingInitializationQueryObservation):
            # No non-personal constraints for initialization query.
            return constraints
        
        if isinstance(obs, FeedingOcclusionQueryObservation):
            plate_position, drink_position = variables

            # The plate and drink cannot be in collision.
            def _plate_drink_collision_free(
                plate_position: NDArray[np.float32],
                drink_position: NDArray[np.float32],
            ) -> bool:
                new_plate_pose = _plate_position_to_pose(plate_position, obs.plate_pose)
                new_drink_pose = _drink_position_to_pose(drink_position, obs.drink_pose)
                set_pose(self._sim.plate_id, new_plate_pose, self._sim.physics_client_id)
                set_pose(self._sim.drink_id, new_drink_pose, self._sim.physics_client_id)
                return not check_body_collisions(
                    self._sim.plate_id, self._sim.drink_id, self._sim.physics_client_id
                )

            plate_drink_collision_free_constraint = FunctionalCSPConstraint(
                "plate_drink_collision_free",
                [plate_position, drink_position],
                _plate_drink_collision_free,
            )
            constraints.append(plate_drink_collision_free_constraint)

            # The plate must be behind the drink.
            def _plate_behind_drink(
                plate_position: NDArray[np.float32],
                drink_position: NDArray[np.float32],
            ) -> bool:
                plate_pos = _plate_position_to_pose(plate_position, obs.plate_pose).position
                drink_pos = _drink_position_to_pose(drink_position, obs.drink_pose).position
                return plate_pos[0] < drink_pos[0]
            
            plate_behind_drink = FunctionalCSPConstraint(
                "plate_behind_drink",
                [plate_position, drink_position],
                _plate_behind_drink,
            )
            constraints.append(plate_behind_drink)

        return constraints

    def _generate_exploit_cost(
        self,
        obs: FeedingObservation,
        variables: list[CSPVariable],
    ) -> CSPCost | None:
        return None

    def _generate_samplers(
        self,
        obs: FeedingObservation,
        csp: CSP,
    ) -> list[CSPSampler]:

        samplers = []

        if isinstance(obs, FeedingInitializationQueryObservation):
            for model, variable in zip([self._feeding_side_model, self._bite_ordering_model, self._ready_signal_model, self._be_verbal_model], csp.variables, strict=True):
                assert model.name == variable.name
                sampler = model.create_sampler(obs, variable, csp)
                samplers.append(sampler)

        elif isinstance(obs, FeedingOcclusionQueryObservation):
            plate_position, drink_position = csp.variables

            def _sample_plate_position(
                _: dict[CSPVariable, Any], rng: np.random.Generator
            ) -> dict[CSPVariable, Any]:
                max_dx = self._sim.scene_spec.table_half_extents[0] - self._sim.scene_spec.plate_radius
                max_dy = self._sim.scene_spec.table_half_extents[1] - self._sim.scene_spec.plate_radius
                dx = rng.uniform(-max_dx, max_dx)
                dy = rng.uniform(-max_dy, max_dy)
                origin = self._sim.scene_spec.table_pose.position[:2]
                new_pos = np.array(
                    [
                        origin[0] + dx,
                        origin[1] + dy,
                    ]
                ).astype(np.float32)

                # visualize sample by adding a small red sphere (alpha=0.5) on pybullet
                viz_pose = Pose(
                    (new_pos[0], new_pos[1], self._sim.scene_spec.table_pose.position[2]),
                    (0, 0, 0, 1),
                )
                self._sim.visualize_sample(viz_pose, color=(1, 0, 0, 0.5))
                return {plate_position: new_pos}

            plate_position_sampler = FunctionalCSPSampler(
                _sample_plate_position, csp, {plate_position}
            )
            samplers.append(plate_position_sampler)
            
            def _sample_drink_position(
                _: dict[CSPVariable, Any], rng: np.random.Generator
            ) -> dict[CSPVariable, Any]:
                max_dx = self._sim.scene_spec.table_half_extents[0] - self._sim.scene_spec.drink_radius
                max_dy = self._sim.scene_spec.table_half_extents[1] - self._sim.scene_spec.drink_radius
                dx = rng.uniform(-max_dx, max_dx)
                dy = rng.uniform(-max_dy, max_dy)
                origin = self._sim.scene_spec.table_pose.position[:2]
                new_pos = np.array(
                    [
                        origin[0] + dx,
                        origin[1] + dy,
                    ]
                ).astype(np.float32)
                # visualize sample by adding a small green sphere (alpha=0.5) on pybullet
                viz_pose = Pose(
                    (new_pos[0], new_pos[1], self._sim.scene_spec.table_pose.position[2]),
                    (0, 0, 0, 1),
                )
                self._sim.visualize_sample(viz_pose, color=(0, 1, 0, 0.5))
                return {drink_position: new_pos}

            drink_position_sampler = FunctionalCSPSampler(
                _sample_drink_position, csp, {drink_position}
            )
            samplers.append(drink_position_sampler)

        return samplers

    def _generate_policy(
        self,
        obs: FeedingObservation,
        csp_variables: Collection[CSPVariable],
    ) -> CSPPolicy:
        return _FeedingCSPPolicy(self._sim, csp_variables, self._seed)

    def observe_transition(
        self,
        obs: FeedingObservation,
        act: FeedingAction,
        next_obs: FeedingObservation,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        
        if isinstance(next_obs, FeedingInitializationDatasetObservation):
            for model in [self._feeding_side_model, self._bite_ordering_model, self._ready_signal_model, self._be_verbal_model]:
                model.learn_incremental(next_obs)



# class FeedingCSPGenerator(CSPGenerator[FeedingObservation, FeedingAction]):
#     """Generate CSPs for the feeding environment."""

#     def __init__(
#         self, sim: FeedingEnv, occlusion_scale_model: Threshold1DModel, *args, **kwargs
#     ) -> None:
#         self._sim = sim
#         self._occlusion_model = occlusion_scale_model
#         super().__init__(*args, **kwargs)

#     def save(self, model_dir: Path) -> None:
#         print("WARNING: saving not yet implemented for FeedingCSPGenerator.")

#     def load(self, model_dir: Path) -> None:
#         print("WARNING: loading not yet implemented for FeedingCSPGenerator.")

#     def _generate_variables(
#         self,
#         obs: FeedingObservation,
#     ) -> tuple[list[CSPVariable], dict[CSPVariable, Any]]:

#         # XY position of the plate.
#         plate_position_domain = Box(
#             np.array([-np.inf, -np.inf]),
#             np.array([np.inf, np.inf]),
#             dtype=np.float32,
#         )
#         plate_position = CSPVariable("plate_position", plate_position_domain)
#         init_plate_position = (obs.plate_pose.position[0], obs.plate_pose.position[1])

#         # XY position of the drink.
#         drink_position_domain = Box(
#             np.array([-np.inf, -np.inf]),
#             np.array([np.inf, np.inf]),
#             dtype=np.float32,
#         )
#         drink_position = CSPVariable("drink_position", drink_position_domain)
#         init_drink_position = (obs.drink_pose.position[0], obs.drink_pose.position[1])

#         return [plate_position, drink_position], {
#             plate_position: init_plate_position,
#             drink_position: init_drink_position,
#         }

#     def _generate_personal_constraints(
#         self,
#         obs: FeedingObservation,
#         variables: list[CSPVariable],
#     ) -> list[CSPConstraint]:

#         constraints: list[CSPConstraint] = []
#         plate_position, drink_position = variables

#         # NOTE: we are currently just using the MLE occlusion scale, rather than
#         # using the full distribution. That means that "ours" will be equivalent
#         # to "exploit_only". This is because we're not really running full
#         # experiments in this environment.

#         # TODO change back!!!!!!!!
#         occlusion_scale = (
#             1.0 - (self._occlusion_model.post_max + self._occlusion_model.post_min) / 2
#         )
#         # occlusion_scale = 0.999
#         self._sim.set_occlusion_scale(occlusion_scale)
#         logging.info(f"Set sim occlusion scale to {occlusion_scale:.3f}")

#         def _user_view_unoccluded_by_utensil(
#             plate_position: NDArray[np.float32],
#         ) -> bool:
#             self._sim.sync_from_observation(obs)
#             new_plate_pose = _plate_position_to_pose(plate_position, obs.plate_pose)
#             field_name = "above_plate_pos"
#             try:
#                 robot_joints = _transform_joints_relative_to_plate(
#                     field_name,
#                     new_plate_pose,
#                     self._sim.robot,
#                     self._sim.scene_spec,
#                     arm_joints_only=False,
#                 )
#             except InverseKinematicsError:
#                 print("WARNING: IK failed within _user_view_unoccluded_by_utensil()")
#                 # from pybullet_helpers.gui import visualize_pose
#                 # visualize_pose(new_plate_pose, self._sim.physics_client_id)
#                 return False
#             held_object_id = self._sim.get_object_id_from_name("utensil")
#             held_object_tf = self._sim.scene_spec.utensil_held_object_tf
#             set_robot_joints_with_held_object(
#                 self._sim.robot,
#                 self._sim.physics_client_id,
#                 held_object_id,
#                 held_object_tf,
#                 robot_joints,
#             )
#             self._sim.robot.set_finger_state(
#                 self._sim.scene_spec.tool_grasp_fingers_value
#             )
#             return not self._sim.robot_in_occlusion()

#         user_view_unoccluded_by_utensil_constraint = FunctionalCSPConstraint(
#             "user_view_unoccluded_by_utensil",
#             [plate_position],
#             _user_view_unoccluded_by_utensil,
#         )

#         if obs.user_request not in ("drink", "prepare-drink-only"):
#             constraints.append(user_view_unoccluded_by_utensil_constraint)

#         def _user_view_unoccluded_by_drink(
#             drink_position: NDArray[np.float32],
#         ) -> bool:
#             self._sim.sync_from_observation(obs)
#             new_drink_pose = _drink_position_to_pose(drink_position, obs.drink_pose)
#             drink_post_grasp_pose = _transform_pose_relative_to_drink(
#                 "drink_default_post_grasp_pose", new_drink_pose, self._sim.scene_spec
#             )
#             # from pybullet_helpers.gui import visualize_pose
#             # visualize_pose(new_drink_pose, self._sim.physics_client_id)
#             try:
#                 robot_joints = inverse_kinematics(
#                     self._sim.robot, drink_post_grasp_pose
#                 )
#             except InverseKinematicsError:
#                 print("WARNING: IK failed within _user_view_unoccluded_by_drink()")
#                 return False
#             held_object_id = self._sim.get_object_id_from_name("drink")
#             held_object_tf = self._sim.scene_spec.drink_held_object_tf
#             set_robot_joints_with_held_object(
#                 self._sim.robot,
#                 self._sim.physics_client_id,
#                 held_object_id,
#                 held_object_tf,
#                 robot_joints,
#             )
#             self._sim.robot.set_finger_state(
#                 self._sim.scene_spec.tool_grasp_fingers_value
#             )
#             return not self._sim.robot_in_occlusion()

#         user_view_unoccluded_by_drink_constraint = FunctionalCSPConstraint(
#             "user_view_unoccluded_by_drink",
#             [drink_position],
#             _user_view_unoccluded_by_drink,
#         )

#         if obs.user_request != "food":
#             constraints.append(user_view_unoccluded_by_drink_constraint)

#         return constraints

#     def _generate_nonpersonal_constraints(
#         self,
#         obs: FeedingObservation,
#         variables: list[CSPVariable],
#     ) -> list[CSPConstraint]:

#         constraints: list[CSPConstraint] = []

#         plate_position, drink_position = variables

#         # # The plate position must be valid w.r.t. IK.
#         # def _plate_position_is_kinematically_valid(
#         #     plate_position: NDArray[np.float32],
#         # ) -> bool:
#         #     new_plate_pose = _plate_position_to_pose(plate_position, obs.plate_pose)
#         #     for field_name in ["before_transfer_pos", "above_plate_pos"]:
#         #         try:
#         #             _transform_joints_relative_to_plate(
#         #                 field_name,
#         #                 new_plate_pose,
#         #                 self._sim.robot,
#         #                 self._sim.scene_spec,
#         #             )
#         #         except InverseKinematicsError:
#         #             return False
#         #     return True

#         # plate_position_kinematically_valid_constraint = FunctionalCSPConstraint(
#         #     "plate_position_kinematically_valid",
#         #     [plate_position],
#         #     _plate_position_is_kinematically_valid,
#         # )

#         # if obs.user_request != "drink":
#         #     constraints.append(plate_position_kinematically_valid_constraint)

#         # The plate and drink cannot be in collision.
#         def _plate_drink_collision_free(
#             plate_position: NDArray[np.float32],
#             drink_position: NDArray[np.float32],
#         ) -> bool:
#             new_plate_pose = _plate_position_to_pose(plate_position, obs.plate_pose)
#             new_drink_pose = _drink_position_to_pose(drink_position, obs.drink_pose)
#             set_pose(self._sim.plate_id, new_plate_pose, self._sim.physics_client_id)
#             set_pose(self._sim.drink_id, new_drink_pose, self._sim.physics_client_id)
#             return not check_body_collisions(
#                 self._sim.plate_id, self._sim.drink_id, self._sim.physics_client_id
#             )

#         plate_drink_collision_free_constraint = FunctionalCSPConstraint(
#             "plate_drink_collision_free",
#             [plate_position, drink_position],
#             _plate_drink_collision_free,
#         )

#         # the plate cannot be too far from the robot base.
#         def _plate_position_reachable(
#             plate_position: NDArray[np.float32],
#         ) -> bool:
#             new_plate_pose = _plate_position_to_pose(plate_position, obs.plate_pose)
#             plate_pos = new_plate_pose.position[:2]
#             print(f"plate is at a distance of {np.linalg.norm(plate_pos)}")
#             return np.linalg.norm(plate_pos) < 0.65
        
#         plate_position_reachable_constraint = FunctionalCSPConstraint(
#             "plate_position_reachable",
#             [plate_position],
#             _plate_position_reachable,
#         )

#         # the drink cannot be too far from the robot base.
#         def _drink_position_reachable(
#             drink_position: NDArray[np.float32],
#         ) -> bool:
#             new_drink_pose = _drink_position_to_pose(drink_position, obs.drink_pose)
#             drink_pos = new_drink_pose.position[:2]
#             print(f"drink is at a distance of {np.linalg.norm(drink_pos)}")
#             return np.linalg.norm(drink_pos) < 0.8 and drink_pos[0] > 0.5 # Not too near user
        
#         drink_position_reachable_constraint = FunctionalCSPConstraint(
#             "drink_position_reachable",
#             [drink_position],
#             _drink_position_reachable,
#         )

#         # the plate must be behind the drink.
#         def _plate_behind_drink(
#             plate_position: NDArray[np.float32],
#             drink_position: NDArray[np.float32],
#         ) -> bool:
#             plate_pos = _plate_position_to_pose(plate_position, obs.plate_pose).position
#             drink_pos = _drink_position_to_pose(drink_position, obs.drink_pose).position
#             print(f"plate is at {plate_pos} and drink is at {drink_pos}")
#             return plate_pos[0] < drink_pos[0]
        
#         plate_behind_drink = FunctionalCSPConstraint(
#             "plate_behind_drink",
#             [plate_position, drink_position],
#             _plate_behind_drink,
#         )

#         if obs.user_request != "drink":
#             constraints.append(plate_drink_collision_free_constraint)
#             # constraints.append(plate_position_reachable_constraint)
#             # constraints.append(drink_position_reachable_constraint)
#             # constraints.append(plate_behind_drink)
#         else:
#             constraints.append(drink_position_reachable_constraint)

#         if obs.user_request != "food":
#             constraints.append(plate_behind_drink)

#         return constraints

#     def _generate_exploit_cost(
#         self,
#         obs: FeedingObservation,
#         variables: list[CSPVariable],
#     ) -> CSPCost | None:
#         return None

#     def _generate_samplers(
#         self,
#         obs: FeedingObservation,
#         csp: CSP,
#     ) -> list[CSPSampler]:

#         # Sample plate positions.
#         plate_position, drink_position = csp.variables

#         samplers = []

#         def _sample_plate_position(
#             _: dict[CSPVariable, Any], rng: np.random.Generator
#         ) -> dict[CSPVariable, Any]:
#             max_dx = self._sim.scene_spec.table_half_extents[0] - self._sim.scene_spec.plate_radius
#             max_dy = self._sim.scene_spec.table_half_extents[1] - self._sim.scene_spec.plate_radius
#             dx = rng.uniform(-max_dx, max_dx)
#             dy = rng.uniform(-max_dy, max_dy)
#             origin = self._sim.scene_spec.table_pose.position[:2]
#             new_pos = np.array(
#                 [
#                     origin[0] + dx,
#                     origin[1] + dy,
#                 ]
#             ).astype(np.float32)

#             # visualize sample by adding a small red sphere (alpha=0.5) on pybullet
#             viz_pose = Pose(
#                 (new_pos[0], new_pos[1], self._sim.scene_spec.table_pose.position[2]),
#                 (0, 0, 0, 1),
#             )
#             self._sim.visualize_sample(viz_pose, color=(1, 0, 0, 0.5))
#             return {plate_position: new_pos}

#         plate_position_sampler = FunctionalCSPSampler(
#             _sample_plate_position, csp, {plate_position}
#         )
#         if obs.user_request != "prepare-drink-only":
#             samplers.append(plate_position_sampler)

#         def _sample_drink_position(
#             _: dict[CSPVariable, Any], rng: np.random.Generator
#         ) -> dict[CSPVariable, Any]:
#             max_dx = self._sim.scene_spec.table_half_extents[0] - self._sim.scene_spec.drink_radius
#             max_dy = self._sim.scene_spec.table_half_extents[1] - self._sim.scene_spec.drink_radius
#             dx = rng.uniform(-max_dx, max_dx)
#             dy = rng.uniform(-max_dy, max_dy)
#             origin = self._sim.scene_spec.table_pose.position[:2]
#             new_pos = np.array(
#                 [
#                     origin[0] + dx,
#                     origin[1] + dy,
#                 ]
#             ).astype(np.float32)
#             # visualize sample by adding a small green sphere (alpha=0.5) on pybullet
#             viz_pose = Pose(
#                 (new_pos[0], new_pos[1], self._sim.scene_spec.table_pose.position[2]),
#                 (0, 0, 0, 1),
#             )
#             self._sim.visualize_sample(viz_pose, color=(0, 1, 0, 0.5))
#             return {drink_position: new_pos}

#         drink_position_sampler = FunctionalCSPSampler(
#             _sample_drink_position, csp, {drink_position}
#         )
#         samplers.append(drink_position_sampler)

#         return samplers

#     def _generate_policy(
#         self,
#         obs: FeedingObservation,
#         csp_variables: Collection[CSPVariable],
#     ) -> CSPPolicy:
#         return _FeedingCSPPolicy(self._sim, csp_variables, self._seed)

#     def observe_transition(
#         self,
#         obs: FeedingObservation,
#         act: FeedingAction,
#         next_obs: FeedingObservation,
#         done: bool,
#         info: dict[str, Any],
#     ) -> None:
#         above_plate_pos = _transform_joints_relative_to_plate(
#             "above_plate_pos", obs.plate_pose, self._sim.robot, self._sim.scene_spec
#         )
#         # When we do real experiments, we will decide whether to take natural
#         # language here and detect whether it's feedback about occlusion, or
#         # to keep it simple we might just keep it binary (occluding or not).
#         if next_obs.user_feedback == "You're blocking my view!":
#             label = True
#         # Positive examples are collected when the robot is at the above plate
#         # position and no negative feedback is given.
#         elif isinstance(act, MoveToJointPositions) and np.allclose(
#             act.joint_positions, above_plate_pos
#         ):
#             label = False
#         else:
#             return
#         self._sim.sync_from_observation(next_obs)
#         occlusion_score = self._sim.get_occlusion_score()
#         self._occlusion_model.fit_incremental([occlusion_score], [label])
#         print(f"Updated occlusion model with {occlusion_score}, {label}")
#         print(f"New params: {self._occlusion_model.get_summary()}")


def _plate_position_to_pose(
    plate_position: NDArray[np.float32], default_pose: Pose
) -> Pose:
    return Pose(
        (
            plate_position[0],
            plate_position[1],
            default_pose.position[2],
        ),
        default_pose.orientation,
    )


def _drink_position_to_pose(
    drink_position: NDArray[np.float32], default_pose: Pose
) -> Pose:
    return Pose(
        (
            drink_position[0],
            drink_position[1],
            default_pose.position[2],
        ),
        default_pose.orientation,
    )


def _transform_joints_relative_to_plate(
    scene_spec_field: str,
    plate_pose: Pose,
    sim_robot: FingeredSingleArmPyBulletRobot,
    scene_spec: FeedingSceneSpec,
    arm_joints_only: bool = True,
) -> JointPositions:
    return _transform_joints_relative_to_default(
        scene_spec_field,
        "plate_default_pose",
        plate_pose,
        sim_robot,
        scene_spec,
        arm_joints_only=arm_joints_only,
    )


def _transform_joints_relative_to_drink(
    scene_spec_field: str,
    drink_pose: Pose,
    sim_robot: FingeredSingleArmPyBulletRobot,
    scene_spec: FeedingSceneSpec,
    arm_joints_only: bool = True,
) -> JointPositions:
    return _transform_joints_relative_to_default(
        scene_spec_field,
        "drink_default_pose",
        drink_pose,
        sim_robot,
        scene_spec,
        arm_joints_only=arm_joints_only,
    )


def _transform_joints_relative_to_default(
    scene_spec_field: str,
    default_scene_field: str,
    pose: Pose,
    sim_robot: FingeredSingleArmPyBulletRobot,
    scene_spec: FeedingSceneSpec,
    arm_joints_only: bool = True,
) -> JointPositions:
    default_positions = getattr(scene_spec, scene_spec_field)
    world_to_default: Pose = getattr(scene_spec, default_scene_field)
    full_joints = sim_robot.get_joint_positions()
    num_dof = len(default_positions)
    full_joints[:num_dof] = default_positions
    sim_robot.set_joints(full_joints)
    world_to_ee = sim_robot.get_end_effector_pose()
    plate_to_ee = world_to_default.invert().multiply(world_to_ee)
    new_ee = pose.multiply(plate_to_ee)
    new_full_joints = inverse_kinematics(sim_robot, new_ee)
    if arm_joints_only:
        return new_full_joints[:num_dof]
    return new_full_joints


def _transform_pose_relative_to_plate(
    scene_spec_field: str, plate_pose: Pose, scene_spec: FeedingSceneSpec
) -> Pose:
    return _transform_pose_relative_to_default(
        scene_spec_field, "plate_default_pose", plate_pose, scene_spec
    )


def _transform_pose_relative_to_drink(
    scene_spec_field: str, drink_pose: Pose, scene_spec: FeedingSceneSpec
) -> Pose:
    return _transform_pose_relative_to_default(
        scene_spec_field, "drink_default_pose", drink_pose, scene_spec
    )


def _transform_pose_relative_to_default(
    pose_scene_spec_field: str,
    default_scene_field: str,
    pose: Pose,
    scene_spec: FeedingSceneSpec,
) -> Pose:
    world_to_pose: Pose = getattr(scene_spec, pose_scene_spec_field)
    world_to_default: Pose = getattr(scene_spec, default_scene_field)
    plate_to_pose = world_to_default.invert().multiply(world_to_pose)
    new_pose = pose.multiply(plate_to_pose)
    return new_pose
