"""Hacky FEAST integration."""

from multitask_personalization.envs.feeding.feeding_env import FeedingEnv, FeedingObservation, BANISH_POSE
from multitask_personalization.envs.feeding.feeding_scene_spec import FeedingSceneSpec, create_feeding_scene_description_from_config
from multitask_personalization.envs.feeding.feeding_csp import _plate_position_to_pose, _drink_position_to_pose, _transform_joints_relative_to_plate, _transform_joints_relative_to_drink, _transform_pose_relative_to_drink, _transform_pose_relative_to_plate
from multitask_personalization.envs.feeding.feeding_structs import FeedingPlateDrinkAction, FeedingOcclusionDatasetObservation, FeedingOcclusionQueryObservation, FeedingInitializationQueryObservation, FeedingInitializationDatasetObservation, FeedingInitializationAction
from multitask_personalization.methods.csp_approach import CSPApproach
from multitask_personalization.csp_solvers import RandomWalkCSPSolver
from pybullet_helpers.geometry import Pose, set_pose
from tomsutils.llm import OpenAILLM
from typing import Any
from pathlib import Path
import cv2


class MultitaskPersonalizationFeastInterface:
    
    def __init__(self, use_gui: bool, personalize: bool = True) -> None:
        self._seed = 0
        self._use_gui = use_gui
        self._personalize = personalize

    def initialize_env(self, config: str) -> None:
        config_file = Path(__file__).parent / "envs" / "feeding" / "configs" / f"{config}.yaml"

        # Create "environment".
        self._scene_spec = create_feeding_scene_description_from_config(config_file)

        # Create approach.
        csp_solver = RandomWalkCSPSolver(self._seed)
        llm = OpenAILLM("gpt-4o-mini", Path("feast_llm_cache"))
        if self._personalize:
            explore_method = "exploit-only"
        else:
            explore_method = "nothing-personal"
        self._approach = CSPApproach(self._scene_spec, None,
                                     csp_solver=csp_solver,
                                     llm=llm,
                                     explore_method=explore_method,
                                     use_gui=self._use_gui)
        self._approach.train()
        self._viz_sim = FeedingEnv(self._scene_spec)

        # Keep track of most recent things that are context for learned constraints.
        self._current_context = None
        self._current_table_type = None
        self._current_food_items = None
        self._current_dips = None
        self._current_bite_ordering_options = None

    def delete_env(self) -> None:
        if hasattr(self, "_viz_sim"):
            self._viz_sim.close()
            del self._viz_sim
        if hasattr(self, "_scene_spec"):
            del self._scene_spec
        if hasattr(self, "_approach"):
            del self._approach

    def run(self, request_dict: dict[str, Any]) -> dict[str, Any] | None:

        if request_dict["request_type"] == "initialization_query":
            
            input("Press ENTER to delete env")
            self.delete_env()

            meal_id = request_dict["meal_id"]
            context = request_dict["context"]
            table_type = request_dict["table_type"]
            food_items = request_dict["food_items"]
            dips = request_dict["dips"]
            bite_ordering_options = request_dict["bite_ordering_options"]

            input("Press ENTER to create env")
            # Initialize the environment with the meal ID.
            self.initialize_env(f"meal_{meal_id}")

            # Save these to use in initialization_dataset below.
            self._current_context = context
            self._current_table_type = table_type
            self._current_food_items = food_items
            self._current_dips = dips
            self._current_bite_ordering_options = bite_ordering_options

            obs = FeedingInitializationQueryObservation(context, table_type, food_items, dips, bite_ordering_options)
            self._approach.reset(obs, {})
            act = self._approach.step()
            assert isinstance(act, FeedingInitializationAction)

            feeding_side = act.feeding_side
            bite_ordering = act.bite_ordering
            ready_signal = act.ready_signal
            be_verbal = act.be_verbal

            return {
                "response_type": "initialization_query",
                "feeding_side": feeding_side,
                "bite_ordering": bite_ordering,
                "ready_signal": ready_signal,
                "be_verbal": be_verbal,
            }
        
        elif request_dict["request_type"] == "initialization_dataset":
            feeding_side = request_dict["feeding_side"]
            bite_ordering = request_dict["bite_ordering"]
            ready_signal = request_dict["ready_signal"]
            be_verbal = request_dict["be_verbal"]

            obs = FeedingInitializationDatasetObservation(
                self._current_context,
                self._current_table_type,
                self._current_food_items,
                self._current_dips,
                self._current_bite_ordering_options,
                feeding_side,
                bite_ordering,
                ready_signal,
                be_verbal
            )
            self._approach.update(obs, 0.0, False, {})

            return {"response_type": "initialization_dataset"}

        elif request_dict["request_type"] == "occlusion_query":

            plate_pose = request_dict["plate_pose"]
            drink_pose = request_dict["drink_pose"]

            self._visualize("Current Scene", plate_pose, drink_pose)

            obs = FeedingOcclusionQueryObservation(
                self._current_context,
                self._current_table_type,
                self._current_food_items,
                self._current_dips,
                self._current_bite_ordering_options,
                plate_pose=plate_pose,
                drink_pose=drink_pose,
            )
            self._approach.update(obs, 0.0, False, {})
            act = self._approach.step()
            assert isinstance(act, FeedingPlateDrinkAction)

            plate_delta_xy = act.plate_delta_xy
            drink_delta_xy = act.drink_delta_xy
            before_transfer_pose = act.before_transfer_pose
            before_transfer_pos = act.before_transfer_pos
            above_plate_pos = act.above_plate_pos
            drink_grasp_pos = act.drink_grasp_pos
            occlusion_poi_relevance = act.occlusion_poi_relevance

            new_plate_pose = Pose((plate_pose.position[0] + plate_delta_xy[0],
                                plate_pose.position[1] + plate_delta_xy[1],
                                plate_pose.position[2]),
                                plate_pose.orientation)
            new_drink_pose = Pose((drink_pose.position[0] + drink_delta_xy[0],
                                drink_pose.position[1] + drink_delta_xy[1],
                                drink_pose.position[2]),
                                drink_pose.orientation)
            
            bite_occlusion_image =  self._render_occlusion_image(new_plate_pose, new_drink_pose, above_plate_pos)
            drink_occlusion_image = self._render_occlusion_image(new_plate_pose, new_drink_pose, drink_grasp_pos)
            # self._visualize("Predicted Scene", new_plate_pose, new_drink_pose)

            return {
                "response_type": "occlusion_query",
                "plate_delta_xy": plate_delta_xy,
                "drink_delta_xy": drink_delta_xy,
                "before_transfer_pose": before_transfer_pose,
                "before_transfer_pos": before_transfer_pos,
                "above_plate_pos": above_plate_pos,
                "drink_grasp_pos": drink_grasp_pos,
                "occlusion_poi_relevance": occlusion_poi_relevance,
                "bite_occlusion_image": bite_occlusion_image,
                "drink_occlusion_image": drink_occlusion_image,
            }
        
        elif request_dict["request_type"] == "occlusion_dataset":

            plate_pose = request_dict["plate_pose"]
            drink_pose = request_dict["drink_pose"]
            occlusion = request_dict["occlusion"]

            obs = FeedingOcclusionDatasetObservation(
                self._current_context,
                self._current_table_type,
                self._current_food_items,
                self._current_dips,
                self._current_bite_ordering_options,
                plate_pose,
                drink_pose,
                occlusion,
            )
            self._approach.update(obs, 0.0, False, {})

            return {"response_type": "occlusion_dataset"}

        else:
            raise ValueError(f"Unknown request type: {request_dict['requestType']}")
        
    def _render_occlusion_image(self, plate_pose: Pose, drink_pose: Pose, robot_joints: list[float]) -> None:
        # Set the plate and drink poses in the simulation.
        set_pose(self._viz_sim.get_object_id_from_name("plate"), plate_pose, self._viz_sim.physics_client_id)
        set_pose(self._viz_sim.get_object_id_from_name("drink"), drink_pose, self._viz_sim.physics_client_id)

        print("robot joints", robot_joints)
        self._viz_sim.robot.set_joints(robot_joints + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Render the image.
        img = self._viz_sim.render(user_view=True)
        return img

    def _visualize(self, title: str, plate_pose: Pose, drink_pose: Pose) -> None:
        set_pose(self._viz_sim.get_object_id_from_name("plate"), plate_pose, self._viz_sim.physics_client_id)
        set_pose(self._viz_sim.get_object_id_from_name("drink"), drink_pose, self._viz_sim.physics_client_id)
        img = self._viz_sim.render()
        from PIL import Image
        print(f"Showing {title}")
        Image.fromarray(img).show()
        input("Press enter to continue")



if __name__ == "__main__":
    import argparse
    import rospy
    import pickle
    from std_msgs.msg import String
    import base64

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_gui",
        action="store_true",
        help="Use GUI.",
    )
    parser.add_argument(
        "--no_personalize",
        action="store_true",
        help="Personalize.",
    )
    args = parser.parse_args()

    interface = MultitaskPersonalizationFeastInterface(args.use_gui, not args.no_personalize)

    def callback(msg):
        request = pickle.loads(base64.b64decode(msg.data))  # convert ByteMultiArray back to object
        # print("Received request:", request)
        response = interface.run(request)
        # print("Sending response:", response)
        msg = String()
        ps = pickle.dumps(response)
        s = base64.b64encode(ps).decode('ascii')
        msg.data = s
        pub.publish(msg)

    rospy.init_node("multitask_personalization_feast_interface")
    sub = rospy.Subscriber('/mp_request', String, callback)
    pub = rospy.Publisher('/mp_response', String, queue_size=1)

    rospy.spin()
