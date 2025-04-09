"""Hacky FEAST integration."""

from multitask_personalization.envs.feeding.feeding_env import FeedingEnv, FeedingState, BANISH_POSE
from multitask_personalization.envs.feeding.feeding_scene_spec import FeedingSceneSpec
from multitask_personalization.envs.feeding.feeding_csp import _plate_position_to_pose, _drink_position_to_pose, _transform_joints_relative_to_plate, _transform_joints_relative_to_drink, _transform_pose_relative_to_drink, _transform_pose_relative_to_plate
from multitask_personalization.methods.csp_approach import CSPApproach
from multitask_personalization.csp_solvers import RandomWalkCSPSolver
from pybullet_helpers.geometry import Pose
from pybullet_helpers.gui import visualize_pose
from typing import Any


class MultitaskPersonalizationFeastInterface:
    
    def __init__(self, personalize: bool = True) -> None:
        self._seed = 0

        # Create "environment".
        self._scene_spec = FeedingSceneSpec()
        self._env = FeedingEnv(self._scene_spec, seed=self._seed, use_gui=False)

        # Create approach.
        csp_solver = RandomWalkCSPSolver(self._seed)
        assert personalize, "TODO"
        explore_method = "exploit-only"
        self._approach = CSPApproach(self._scene_spec, self._env.action_space,
                                     csp_solver=csp_solver,
                                     explore_method=explore_method)
        self._approach.train()
        
    def run(self, feast_state_dict: dict[str, Any]) -> dict[str, Any]:

        sim_state = self._env.get_state()

        robot_joints = feast_state_dict["robot_joints"]

        detected_plate_pose = feast_state_dict["plate_pose"]
        plate_pose = Pose(
            (detected_plate_pose.position[0],
             detected_plate_pose.position[1],
             sim_state.plate_pose.position[2]),
            detected_plate_pose.orientation
        )
        visualize_pose(plate_pose, self._env.physics_client_id)

        detected_drink_pose = feast_state_dict.get("drink_pose", BANISH_POSE)
        drink_pose = Pose(
            (detected_drink_pose.position[0],
             detected_drink_pose.position[1],
             sim_state.drink_pose.position[2]),
            detected_drink_pose.orientation
        )
        visualize_pose(plate_pose, self._env.physics_client_id)

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
        self._approach.reset(feeding_state, {})

        sol = self._approach._current_sol
        plate_var, drink_var = sol.keys()
        assert "plate" in plate_var.name
        assert "drink" in drink_var.name
        plate_position = sol[plate_var]
        drink_position = sol[drink_var]
        new_plate_pose = _plate_position_to_pose(plate_position, plate_pose)
        new_drink_pose = _drink_position_to_pose(drink_position, drink_pose)

        before_transfer_pose = _transform_pose_relative_to_plate(
            "before_transfer_pose", new_plate_pose, self._env.scene_spec
        )

        before_transfer_pos = _transform_joints_relative_to_plate(
            "before_transfer_pos", new_plate_pose, self._env.robot, self._env.scene_spec
        )

        above_plate_pos = _transform_joints_relative_to_plate(
            "above_plate_pos", new_plate_pose, self._env.robot, self._env.scene_spec
        )

        return {
            "before_transfer_pose": before_transfer_pose,
            "before_transfer_pos": before_transfer_pos,
            "above_plate_pos": above_plate_pos,
        }


if __name__ == "__main__":
    import rospy
    import pickle
    from std_msgs.msg import String
    import base64

    interface = MultitaskPersonalizationFeastInterface()

    def callback(msg):
        obj = pickle.loads(base64.b64decode(msg.data))  # convert ByteMultiArray back to object
        print("Received object:", obj)
        scene_spec_updates = interface.run(obj)
        pub.publish(base64.b64encode(pickle.dumps(scene_spec_updates)))

    rospy.init_node("multitask_personalization_feast_interface")
    sub = rospy.Subscriber('/mp_state', String, callback)
    pub = rospy.Publisher('/mp_state_out', String, queue_size=1)

    rospy.spin()
