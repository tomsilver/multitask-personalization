"""Hacky FEAST integration."""

from multitask_personalization.envs.feeding.feeding_env import FeedingEnv, FeedingState, BANISH_POSE
from multitask_personalization.envs.feeding.feeding_scene_spec import FeedingSceneSpec
from multitask_personalization.methods.csp_approach import CSPApproach
from multitask_personalization.csp_solvers import RandomWalkCSPSolver
from typing import Any


class MultitaskPersonalizationFeastInterface:
    
    def __init__(self, personalize: bool = True) -> None:
        self._seed = 0

        # Create "environment".
        self._scene_spec = FeedingSceneSpec()
        self._env = FeedingEnv(self._scene_spec, seed=self._seed, use_gui=True)
        self._env.reset()

        # Create approach.
        csp_solver = RandomWalkCSPSolver(self._seed)
        assert personalize, "TODO"
        explore_method = "exploit-only"
        self._approach = CSPApproach(self._scene_spec, self._env.action_space,
                                     csp_solver=csp_solver,
                                     explore_method=explore_method)
        self._approach.train()
        
    def run(self, feast_state_dict: dict[str, Any]) -> dict[str, Any]:

        plate_pose = feast_state_dict["plate_pose"]
        drink_pose = feast_state_dict.get("drink_pose", BANISH_POSE)
        robot_joints = feast_state_dict["robot_joints"]
        
        feeding_state = FeedingState(
            robot_joints=robot_joints,
            plate_pose=plate_pose,
            drink_pose=drink_pose,
            held_object_name=None,
            held_object_tf=None,
            stage="acquisition",
            user_request="food",
            user_feedback=None,
        )
        self._env.set_state(feeding_state)
        import ipdb; ipdb.set_trace()
        self._approach.reset(feeding_state, {})
    

if __name__ == "__main__":
    import rospy
    import pickle
    from std_msgs.msg import String
    import base64

    interface = MultitaskPersonalizationFeastInterface()

    def callback(msg):
        obj = pickle.loads(base64.b64decode(msg.data))  # convert ByteMultiArray back to object
        print("Received object:", obj)
        interface.run(obj)

    rospy.init_node("multitask_personalization_feast_interface")
    sub = rospy.Subscriber('/mp_state', String, callback)

    rospy.spin()
