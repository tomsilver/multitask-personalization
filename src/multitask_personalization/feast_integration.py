"""Hacky FEAST integration."""

from multitask_personalization.envs.feeding.feeding_env import FeedingEnv
from multitask_personalization.envs.feeding.feeding_scene_spec import FeedingSceneSpec
from multitask_personalization.envs.feeding.feeding_structs import FeedingPlateDrinkAction, FeedingOcclusionDatasetObservation, FeedingOcclusionQueryObservation, FeedingInitializationQueryObservation, FeedingInitializationDatasetObservation, FeedingInitializationAction
from multitask_personalization.methods.csp_approach import CSPApproach
from multitask_personalization.csp_solvers import RandomWalkCSPSolver
from tomsutils.llm import OpenAILLM
from typing import Any
from pathlib import Path


class MultitaskPersonalizationFeastInterface:
    
    def __init__(self, use_gui: bool, personalize: bool = True) -> None:
        self._seed = 0

        # Create "environment".
        self._scene_spec = FeedingSceneSpec()
        self._env = FeedingEnv(self._scene_spec, seed=self._seed, use_gui=use_gui)

        # Create approach.
        csp_solver = RandomWalkCSPSolver(self._seed)
        llm = OpenAILLM("gpt-4o-mini", Path("feast_llm_cache"))
        if personalize:
            explore_method = "exploit-only"
        else:
            explore_method = "nothing-personal"
        self._approach = CSPApproach(self._scene_spec, self._env.action_space,
                                     csp_solver=csp_solver,
                                     llm=llm,
                                     explore_method=explore_method)
        self._approach.train()

        # Keep track of most recent things that are context for learned constraints.
        self._current_context = None
        self._current_table_type = None
        self._current_food_items = None
        self._current_dips = None
        self._current_bite_ordering_options = None

    def run(self, request_dict: dict[str, Any]) -> dict[str, Any] | None:

        if request_dict["request_type"] == "initialization_query":
            context = request_dict["context"]
            table_type = request_dict["table_type"]
            food_items = request_dict["food_items"]
            dips = request_dict["dips"]
            bite_ordering_options = request_dict["bite_ordering_options"]

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

            return {
                "response_type": "occlusion_query",
                "plate_delta_xy": plate_delta_xy,
                "drink_delta_xy": drink_delta_xy,
                "before_transfer_pose": before_transfer_pose,
                "before_transfer_pos": before_transfer_pos,
                "above_plate_pos": above_plate_pos,
            }
        
        elif request_dict["request_type"] == "occlusion_dataset":

            plate_pose = request_dict["plate_pose"]
            plate_occlusion = request_dict["plate_occlusion"]
            drink_pose = request_dict["drink_pose"]
            drink_occlusion = request_dict["drink_occlusion"]

            obs = FeedingOcclusionDatasetObservation(
                self._current_context,
                self._current_table_type,
                self._current_food_items,
                self._current_dips,
                self._current_bite_ordering_options,
                plate_pose,
                drink_pose,
                plate_occlusion,
                drink_occlusion,
            )
            self._approach.update(obs, 0.0, False, {})

            return {"response_type": "occlusion_dataset"}

        else:
            raise ValueError(f"Unknown request type: {request_dict['requestType']}")

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
        print("Received request:", request)
        response = interface.run(request)
        print("Sending response:", response)
        msg = String()
        ps = pickle.dumps(response)
        s = base64.b64encode(ps).decode('ascii')
        msg.data = s
        pub.publish(msg)

    rospy.init_node("multitask_personalization_feast_interface")
    sub = rospy.Subscriber('/mp_request', String, callback)
    pub = rospy.Publisher('/mp_response', String, queue_size=1)

    rospy.spin()
