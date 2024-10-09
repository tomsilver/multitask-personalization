"""A domain-specific parameterized policy for pybullet handover."""

import numpy as np

from pybullet_helpers.geometry import Pose3D
from pybullet_helpers.link import get_link_pose

from multitask_personalization.envs.pybullet_handover import PyBulletHandoverSceneDescription, PyBulletHandoverSimulator, _HandoverState, _HandoverAction
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
        self, scene_description: PyBulletHandoverSceneDescription, seed: int = 0,
    ) -> None:
        super().__init__()
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        # Create a simulator for planning.
        self._sim = PyBulletHandoverSimulator(scene_description)
        # Store an action plan for the robot.
        self._plan: list[_HandoverAction] = []

    def reset(self, task_id: str, parameters: float) -> None:
        super().reset(task_id, parameters)
        self._plan = []

    def step(self, state: _HandoverState) -> _HandoverAction:
        assert self._current_parameters is not None
        if not self._plan:
            # Sample a handover position on the surface of the current
            # estimated sphere.
            handover_position = self._sample_handover_position(self._current_parameters)
            self._plan = self._get_handover_plan(state, handover_position)
        assert len(self._plan) > 0
        return self._plan.pop(0)

    def _sample_handover_position(self, radius: float) -> Pose3D:
        # Get the sphere center from the simulator.
        center = get_link_pose(
            self._sim.human.body, self._sim.human.right_wrist, self._sim.physics_client_id
        ).position
        return sample_spherical(center, radius, self._rng)

    def _get_handover_plan(self, state: _HandoverState, handover_position: Pose3D) -> list[_HandoverAction]:
        import ipdb; ipdb.set_trace()
    