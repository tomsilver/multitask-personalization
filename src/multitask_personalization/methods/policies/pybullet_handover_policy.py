"""A domain-specific parameterized policy for pybullet handover."""

import numpy as np
from pybullet_helpers.geometry import Pose
from pybullet_helpers.link import get_link_pose
from pybullet_helpers.motion_planning import (
    run_smooth_motion_planning_to_pose,
)

from multitask_personalization.envs.pybullet_handover import (
    PyBulletHandoverSceneDescription,
    PyBulletHandoverSimulator,
    _HandoverAction,
    _HandoverState,
)
from multitask_personalization.methods.policies.parameterized_policy import (
    ParameterizedPolicy,
)
from multitask_personalization.utils import sample_spherical


class PyBulletHandoverParameterizedPolicy(
    ParameterizedPolicy[_HandoverState, _HandoverAction, float]
):
    """A domain-specific parameterized policy for pybullet handover.

    Parameters define the range of motion model. For now, we have a very simple
    ROM model that only has one parameter: the radius of a sphere around the
    person's hand.
    """

    def __init__(
        self,
        scene_description: PyBulletHandoverSceneDescription,
        seed: int = 0,
        max_motion_planning_time: float = 1.0,
    ) -> None:
        super().__init__()
        self._seed = seed
        self._max_motion_planning_time = max_motion_planning_time
        self._rng = np.random.default_rng(seed)
        # Create a simulator for planning.
        self._sim = PyBulletHandoverSimulator(scene_description, use_gui=False)
        # Store an action plan for the robot.
        self._plan: list[_HandoverAction] = []

    def reset(self, task_id: str, parameters: float) -> None:
        super().reset(task_id, parameters)
        self._plan = []

    def step(self, state: _HandoverState) -> _HandoverAction:
        assert self._current_parameters is not None
        if not self._plan:
            # Sample a handover position on the surface of the current
            # estimated sphere. Repeatedly sample until a plan is found.
            while True:
                handover_pose = self._sample_handover_pose(self._current_parameters)
                handover_plan = self._get_handover_plan(state, handover_pose)
                if handover_plan is not None:
                    self._plan = handover_plan
                    break
        assert len(self._plan) > 0
        return self._plan.pop(0)

    def _sample_handover_pose(self, radius: float) -> Pose:
        # Get the sphere center from the simulator.
        center = get_link_pose(
            self._sim.human.body,
            self._sim.human.right_wrist,
            self._sim.physics_client_id,
        ).position
        position = sample_spherical(center, radius, self._rng)
        orientation = self._sim.robot.get_end_effector_pose().orientation
        return Pose(position, orientation)

    def _get_handover_plan(
        self, state: _HandoverState, handover_pose: Pose
    ) -> list[_HandoverAction] | None:
        # Motion plan to the handover pose.
        self._sim.set_state(state)
        pybullet_helpers_plan = run_smooth_motion_planning_to_pose(
            handover_pose,
            self._sim.robot,
            collision_ids=set(),
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=self._seed,
            max_time=self._max_motion_planning_time,
        )
        # This can happen if a handover position that is out of reach for the
        # robot is sampled.
        if pybullet_helpers_plan is None:
            return None

        handover_plan: list[_HandoverAction] = []
        previous_joints = state.robot_joints[:7]
        for joints in pybullet_helpers_plan:
            delta_joints = np.subtract(joints[:7], previous_joints).tolist()
            handover_plan.append((0, delta_joints))
            previous_joints = joints[:7]
        return handover_plan
