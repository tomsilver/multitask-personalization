"""CSP elements for the feeding environment."""

import logging
from pathlib import Path
from typing import Any, Callable, Collection
from functools import partial

import numpy as np
import pickle
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
from multitask_personalization.envs.pybullet.pybullet_utils import (
    BANISH_POSE,
)


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
            plate_delta_xy = (planned_plate_position[0] - obs.plate_pose.position[0],
                              planned_plate_position[1] - obs.plate_pose.position[1])
            planned_plate_pose = _plate_position_to_pose(planned_plate_position, obs.plate_pose)
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
            drink_delta_xy = (planned_drink_position[0] - obs.drink_pose.position[0],
                              planned_drink_position[1] - obs.drink_pose.position[1])
            planned_drink_pose = _drink_position_to_pose(planned_drink_position, obs.drink_pose)
            # Rajat ToDo: change default to a logged pickup pos
            # drink_pickup_pose = planned_drink_pose.multiply(Pose((0.0, 0.0, 0.05), (0.0, 0.0, 0.0, 1.0)))
            if not obs.drink_pose.allclose(BANISH_POSE):
                drink_grasp_pos = _transform_joints_relative_to_drink(
                    "drink_staging_pos", planned_drink_pose, self._sim.robot, self._sim.scene_spec
                )
            else:
                drink_grasp_pos = None
            
            occlusion_poi_relevance = {}
            for poi in self._sim.scene_spec.occlusion_points_of_interest:
                relevance = self._get_value(f"occlusion-poi-{poi}")
                occlusion_poi_relevance[poi] = relevance
            return FeedingPlateDrinkAction(plate_delta_xy=plate_delta_xy,
                                           drink_delta_xy=drink_delta_xy,
                                           before_transfer_pose=before_transfer_pose,
                                           before_transfer_pos=before_transfer_pos,
                                           above_plate_pos=above_plate_pos,
                                           drink_grasp_pos=drink_grasp_pos,
                                           occlusion_poi_relevance=occlusion_poi_relevance)
        raise NotImplementedError

    def check_termination(self, obs: FeedingObservation) -> bool:
        return True
    

class LLMMultipleChoiceConstraintModel:
    """Shared code for table side, bite ordering, etc."""

    def __init__(
        self,
        name: str,
        description: str,
        llm: LargeLanguageModel,
        get_choices_from_observation: Callable[[FeedingObservation], list[Any]],
        get_variable_value_from_choice: Callable[[Any, FeedingObservation], Any] | None = None,
        seed: int = 0,
    ):
        self.name = name  # this must match the name in FeedingInitializationDatasetObservation
        self.description = description
        self.llm = llm
        self.get_choices_from_observation = get_choices_from_observation
        self.get_variable_value_from_choice = get_variable_value_from_choice or (lambda x, o: x)
        self.summary_preferences = "Unknown"
        self.seed = seed
        self.data_obs_history: list[FeedingInitializationDatasetObservation | FeedingOcclusionQueryObservation] = []

    def get_save_state(self) -> dict:
        return {
            "data_obs_history": list(self.data_obs_history),
            "summary_preferences": self.summary_preferences,
        }
    
    def load_from_state(self, state_dict: dict) -> None:
        self.data_obs_history = list(state_dict["data_obs_history"])
        self.summary_preferences = state_dict["summary_preferences"]

    def save(self, model_path: Path) -> None:
        with open(model_path, "wb") as f:
            pickle.dump(self.get_save_state(), f)

    def load(self, model_path: Path) -> None:
        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
                self.load_from_state(data)
        except FileNotFoundError:
            logging.warning(f"Model file {model_path} not found. Using init values.")

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
        # print(prompt)
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
        prompt = f"""You are a mealtime assistance robot and you are choosing a value for a variable `{self.name}`, which means "{self.description}", and which can take on the following values:

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
        self.data_obs_history.append(obs)
        history_str = self.get_history_str()
        prompt = f"""You are a mealtime assistance robot and you are summarizing a user's preferences for a variable `{self.name}`, which means "{self.description}". Here is a history of (state, user choice):

{history_str}

Based on this history, summarize the user's preferences.

IMPORTANT: Use common sense to understand what aspects of the state might be relevant for the variable.
"""

        response, _ = self.llm.query(prompt, temperature=1.0, seed=self.seed)
        self.summary_preferences = response
        print(f"Updated user preferences for {self.name}: {self.summary_preferences}")
    
    def get_history_str(self) -> str:
        combined_str = ""
        for obs in self.data_obs_history:
            if isinstance(obs, FeedingInitializationDatasetObservation):
                user_choice = getattr(obs, self.name)
            else:
                assert isinstance(obs, FeedingOcclusionDatasetObservation)
                assert self.name.startswith("occlusion-poi-")
                poi = self.name[len("occlusion-poi-"):]
                user_choice = obs.occlusion[poi]["relevance"]
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
        self._feeding_side_model = LLMMultipleChoiceConstraintModel("feeding_side", "the side that the robot arm will feed from", llm, lambda o: ["left", "right"])
        self._bite_ordering_model = LLMMultipleChoiceConstraintModel("bite_ordering","the order of food items that the robot will serve", llm,lambda o: o.bite_ordering_options,
                                                                     get_variable_value_from_choice=lambda x,o: o.bite_ordering_options.index(x))
        self._ready_signal_model = LLMMultipleChoiceConstraintModel("ready_signal", "how the robot should indicate ready for food or drink transfer", llm, lambda o: ["mouth_open", "button", "auto_continue"])
        self._be_verbal_model = LLMMultipleChoiceConstraintModel("be_verbal", "whether the robot should speak", llm, lambda o: [True, False])
        self._occlusion_poi_relevance_models = {
            poi: LLMMultipleChoiceConstraintModel(f"occlusion-poi-{poi}", f"whether the user might be looking in the {poi} direction during this meal", llm, lambda o: [True, False])
            for poi in self._sim.scene_spec.occlusion_points_of_interest
        }

    def get_save_state(self) -> dict:
        return {
            "feeding_side_model": self._feeding_side_model.get_save_state(),
            "bite_ordering_model": self._bite_ordering_model.get_save_state(),
            "ready_signal_model": self._ready_signal_model.get_save_state(),
            "be_verbal_model": self._be_verbal_model.get_save_state(),
            "occlusion_poi_relevance_models": {
                poi: model.get_save_state() for poi, model in self._occlusion_poi_relevance_models.items()
            },
            "occlusion_model": self._occlusion_model.get_save_state(),
        }
    
    def load_from_state(self, state_dict: dict) -> None:
        self._feeding_side_model.load_from_state(state_dict["feeding_side_model"])
        self._bite_ordering_model.load_from_state(state_dict["bite_ordering_model"])
        self._ready_signal_model.load_from_state(state_dict["ready_signal_model"])
        self._be_verbal_model.load_from_state(state_dict["be_verbal_model"])
        
        for poi, model_state in state_dict["occlusion_poi_relevance_models"].items():
            self._occlusion_poi_relevance_models[poi].load_from_state(model_state)

        self._occlusion_model.load_from_state(state_dict["occlusion_model"])

    def save(self, model_dir: Path) -> None:
        
        # Save constraint models
        self._feeding_side_model.save(model_dir / "feeding_side.pkl")
        self._bite_ordering_model.save(model_dir / "bite_ordering.pkl")
        self._ready_signal_model.save(model_dir / "ready_signal.pkl")
        self._be_verbal_model.save(model_dir / "be_verbal.pkl")

        for model in self._occlusion_poi_relevance_models.values():
            model.save(model_dir / f"occlusion-poi-{model.name}.pkl")

        # Save occlusion scale model
        occlusion_path = model_dir / "occlusion_model.pkl"
        occlusion_model_state = self._occlusion_model.get_save_state()
        with open(occlusion_path, "wb") as f:
            pickle.dump(occlusion_model_state, f)

    def load(self, model_dir: Path) -> None:

        # Load constraint models
        self._feeding_side_model.load(model_dir / "feeding_side.pkl")
        self._bite_ordering_model.load(model_dir / "bite_ordering.pkl")
        self._ready_signal_model.load(model_dir / "ready_signal.pkl")
        self._be_verbal_model.load(model_dir / "be_verbal.pkl")

        for model in self._occlusion_poi_relevance_models.values():
            model.load(model_dir / f"occlusion-poi-{model.name}.pkl")

        # Load occlusion scale model
        try:
            occlusion_path = model_dir / "occlusion_model.pkl"
            with open(occlusion_path, "rb") as f:
                occlusion_model_state = pickle.load(f)
                self._occlusion_model.load_from_state(occlusion_model_state)
        except FileNotFoundError:
            logging.warning(f"Model file {occlusion_path} not found. Using init values.")

    def close(self) -> None:
        self._sim.close()

    def _generate_variables(
        self,
        obs: FeedingObservation,
    ) -> tuple[list[CSPVariable], dict[CSPVariable, Any]]:
        
        if isinstance(obs, FeedingInitializationQueryObservation):
            print("Generating a CSP for feeding initialization")

            feeding_side = CSPVariable("feeding_side", EnumSpace(["left", "right"]))
            bite_ordering = CSPVariable("bite_ordering", Discrete(len(obs.bite_ordering_options)))  # index into obs.bite_ordering_options
            ready_signal = CSPVariable("ready_signal", EnumSpace(["mouth_open", "button", "auto_continue"]))
            be_verbal = CSPVariable("be_verbal", EnumSpace([True, False]))
            variables = [feeding_side, bite_ordering, ready_signal, be_verbal]

            initialization = {
                feeding_side: self._feeding_side_model.get_most_preferred_choice(obs),
                bite_ordering: self._bite_ordering_model.get_most_preferred_choice(obs),
                ready_signal: self._ready_signal_model.get_most_preferred_choice(obs),
                be_verbal: self._be_verbal_model.get_most_preferred_choice(obs),
            }

            return variables, initialization
        
        if isinstance(obs, FeedingInitializationDatasetObservation):
            return [], {}
        
        if isinstance(obs, FeedingOcclusionQueryObservation):
            print("Generating a CSP for occlusion avoidance")

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
            
            # Whether each of the occlusion points of interest are relevant.
            for point_of_interest in self._sim.scene_spec.occlusion_points_of_interest:
                poi_relevant = CSPVariable(f"occlusion-poi-{point_of_interest}", EnumSpace([True, False]))
                variables.append(poi_relevant)
                initialization[poi_relevant] = self._occlusion_poi_relevance_models[point_of_interest].get_most_preferred_choice(obs)

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

        # Add occlusion constraints.
        if isinstance(obs, FeedingOcclusionQueryObservation):
            plate_position, drink_position = variables[:2]
            occlusion_poi_vars = variables[2:]
            poi_to_occlusion_var = {}
            for occlusion_poi_var in occlusion_poi_vars:
                prefix = "occlusion-poi-"
                assert occlusion_poi_var.name.startswith(prefix)
                poi = occlusion_poi_var.name[len(prefix):]
                poi_to_occlusion_var[poi] = occlusion_poi_var
            
            # NOTE: we are currently just using the MLE occlusion scale, rather than
            # using the full distribution. That means that "ours" will be equivalent
            # to "exploit_only". This is because we're not really running full
            # experiments in this environment.
            occlusion_scale = (
                1.0 - (self._occlusion_model.post_max + self._occlusion_model.post_min) / 2
            )
            if not any(self._occlusion_model.incremental_Y):
                occlusion_scale = 0.0
            # occlusion_scale = 0.999
            occlusion_scale = min(0.99, occlusion_scale)  # cap at 0.99
            print(f"Using occlusion scale {occlusion_scale:.3f}")
            
            def _user_view_unoccluded_by_utensil(
                occlusion_poi: str,
                poi_is_relevant: bool,
                plate_position: NDArray[np.float32],
            ) -> bool:
                if not poi_is_relevant:
                    return True
                score = self._get_plate_occlusion_score(plate_position, occlusion_poi)
                return score is not None and score < 1.0 - occlusion_scale
            
            for poi, occlusion_poi_var in poi_to_occlusion_var.items():
                user_view_unoccluded_by_utensil_constraint = FunctionalCSPConstraint(
                    f"user_view_unoccluded_by_utensil_{poi}",
                    [occlusion_poi_var, plate_position],
                    partial(_user_view_unoccluded_by_utensil, poi),
                )
                constraints.append(user_view_unoccluded_by_utensil_constraint)

            def _user_view_unoccluded_by_drink(
                occlusion_poi: str,
                poi_is_relevant: bool,
                drink_position: NDArray[np.float32],
            ) -> bool:
                if not poi_is_relevant:
                    return True
                score = self._get_drink_occlusion_score(drink_position, occlusion_poi)
                return score is not None and score < 1.0 - occlusion_scale
            
            if not (hasattr(self, "_disable_drink") and self._disable_drink):
                for poi, occlusion_poi_var in poi_to_occlusion_var.items():
                    user_view_unoccluded_by_drink_constraint = FunctionalCSPConstraint(
                        f"user_view_unoccluded_by_drink_{poi}",
                        [occlusion_poi_var, drink_position],
                        partial(_user_view_unoccluded_by_drink, poi),
                    )
                    constraints.append(user_view_unoccluded_by_drink_constraint)
            
            # Relevance constraints.
            for poi, occlusion_poi_var in poi_to_occlusion_var.items():
                model = self._occlusion_poi_relevance_models[poi]
                constraints.append(model.create_constraint(obs, occlusion_poi_var))
            
        return constraints

    def _generate_nonpersonal_constraints(
        self,
        obs: FeedingObservation,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:

        constraints: list[CSPConstraint] = []

        if isinstance(obs, FeedingOcclusionQueryObservation):
            plate_position, drink_position = variables[:2]

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
            if not (hasattr(self, "_disable_drink") and self._disable_drink):
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
            if not (hasattr(self, "_disable_drink") and self._disable_drink):
                constraints.append(plate_behind_drink)

            def _plate_waypoints_reachable(
                plate_position: NDArray[np.float32],
            ) -> bool:
                planned_plate_pose = _plate_position_to_pose(plate_position, obs.plate_pose)
                try:
                    _transform_joints_relative_to_plate(
                        "before_transfer_pos", planned_plate_pose, self._sim.robot, self._sim.scene_spec
                    )
                    _transform_joints_relative_to_plate(
                        "above_plate_pos", planned_plate_pose, self._sim.robot, self._sim.scene_spec,
                    )
                except InverseKinematicsError:
                    return False
                return True
            
            plate_waypoints_reachable_constraint = FunctionalCSPConstraint(
                "plate_waypoints_reachable",
                [plate_position],
                _plate_waypoints_reachable,
            )
            constraints.append(plate_waypoints_reachable_constraint)

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
            plate_position, drink_position = csp.variables[:2]
            occlusion_poi_vars = csp.variables[2:]
            poi_to_occlusion_var = {}
            for occlusion_poi_var in occlusion_poi_vars:
                prefix = "occlusion-poi-"
                assert occlusion_poi_var.name.startswith(prefix)
                poi = occlusion_poi_var.name[len(prefix):]
                poi_to_occlusion_var[poi] = occlusion_poi_var

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
                if not (hasattr(self, "_disable_drink")):
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
            if not (hasattr(self, "_disable_drink") and self._disable_drink):
                samplers.append(drink_position_sampler)

            # Relevance samplers.
            for poi, occlusion_poi_var in poi_to_occlusion_var.items():
                model = self._occlusion_poi_relevance_models[poi]
                sampler = model.create_sampler(obs, occlusion_poi_var, csp)
                samplers.append(sampler)

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
            print("Updating models for feeding initialization")
            for model in [self._feeding_side_model, self._bite_ordering_model, self._ready_signal_model, self._be_verbal_model]:
                model.learn_incremental(next_obs)

        if isinstance(next_obs, FeedingOcclusionDatasetObservation):
            print("Updating models for occlusions")
            X, Y = [], []  # for occlusion model

            for point_of_interest, feedback in next_obs.occlusion.items():
                relevant = feedback["relevance"]
                relevance_model = self._occlusion_poi_relevance_models[point_of_interest]
                relevance_model.learn_incremental(next_obs)
                if relevant:
                    plate_pose = next_obs.plate_pose
                    plate_score = self._get_plate_occlusion_score(plate_pose.position[:2], point_of_interest)
                    assert plate_score is not None, "Shouldn't be possible if IK is checked during constraint solving..."
                    plate_label = feedback["plate_occlusion"]
                    # if plate_label and np.isclose(plate_score, 0.0):
                    #     print("OH NO!!!! We are screwed. User said there was occlusion when our model thinks none is possible.")
                    #     import ipdb; ipdb.set_trace()
                    X.append(plate_score)
                    Y.append(plate_label)

                    if not (hasattr(self, "_disable_drink") and self._disable_drink):
                        drink_pose = next_obs.drink_pose
                        drink_score = self._get_drink_occlusion_score(drink_pose.position[:2], point_of_interest)
                        assert drink_score is not None, "Shouldn't be possible if IK is checked during constraint solving..."
                        drink_label = feedback["drink_occlusion"]
                        # if drink_label and np.isclose(drink_score, 0.0):
                        #     print("OH NO!!!! We are screwed. User said there was occlusion when our model thinks none is possible.")
                        #     import ipdb; ipdb.set_trace()
                        X.append(drink_score)
                        Y.append(drink_label)

            self._occlusion_model.fit_incremental(X, Y)
            print("Updating occlusion model, new scale:", (
                1.0 - (self._occlusion_model.post_max + self._occlusion_model.post_min) / 2
            ))

    def _get_plate_occlusion_score(self, plate_position: NDArray[np.float32], point_of_interest: str) -> float | None:
        set_pose(self._sim.get_object_id_from_name("drink"), BANISH_POSE, self._sim.physics_client_id)
        new_plate_pose = _plate_position_to_pose(plate_position, self._sim.scene_spec.plate_default_pose)
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
            return None
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
        return self._sim.get_occlusion_score(point_of_interest)

    def _get_drink_occlusion_score(self, drink_position: NDArray[np.float32], point_of_interest: str) -> float | None:
        set_pose(self._sim.get_object_id_from_name("utensil"), BANISH_POSE, self._sim.physics_client_id)
        new_drink_pose = _drink_position_to_pose(drink_position, self._sim.scene_spec.drink_default_pose)
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
            return None
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
        return self._sim.get_occlusion_score(point_of_interest)


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
