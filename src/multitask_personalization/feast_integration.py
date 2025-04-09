"""Hacky FEAST integration."""

from multitask_personalization.envs.feeding.feeding_env import FeedingEnv, FeedingState, BANISH_POSE
from multitask_personalization.envs.feeding.feeding_scene_spec import FeedingSceneSpec
from multitask_personalization.methods.csp_approach import CSPApproach
from multitask_personalization.csp_solvers import RandomWalkCSPSolver
from typing import Any


class MultitaskPersonalizationFeastInterface:
    
    def __init__(self, personalize: bool = True) -> None:
        self.seed = 0

        # Create "environment".
        self._scene_spec = FeedingSceneSpec()
        self._env = FeedingEnv(self._scene_spec, seed=self._seed)
        self._env.reset()

        # Create approach.
        csp_solver = RandomWalkCSPSolver(self._seed)
        assert personalize, "TODO"
        explore_method = "exploit_only"
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
        self._approach.reset(feeding_state, {})
        import ipdb; ipdb.set_trace()
    