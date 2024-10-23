"""CSP elements for the PyBullet environment."""

from typing import Any

import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
)
from pybullet_helpers.math_utils import get_poses_facing_line
from tomsutils.spaces import EnumSpace

from multitask_personalization.envs.pybullet.pybullet_env import PyBulletEnv
from multitask_personalization.envs.pybullet.pybullet_skills import (
    get_plan_to_handover_object,
    get_plan_to_pick_object,
)
from multitask_personalization.envs.pybullet.pybullet_structs import (
    PyBulletAction,
    PyBulletState,
)
from multitask_personalization.rom.models import ROMModel
from multitask_personalization.structs import (
    CSP,
    CSPConstraint,
    CSPPolicy,
    CSPSampler,
    CSPVariable,
    FunctionalCSPSampler,
)


class _BookHandoverCSPPolicy(CSPPolicy[PyBulletState, PyBulletAction]):

    def __init__(self, sim: PyBulletEnv, csp: CSP, seed: int = 0) -> None:
        super().__init__(csp, seed)
        self._sim = sim
        self._current_plan: list[PyBulletAction] = []

    def reset(self, solution: dict[CSPVariable, Any]) -> None:
        super().reset(solution)
        self._current_plan = []

    def step(self, obs: PyBulletState) -> PyBulletAction:
        if not self._current_plan:
            if obs.held_object is None:
                self._current_plan = self._get_pick_plan(obs)
            else:
                self._current_plan = self._get_handover_plan(obs)
        return self._current_plan.pop(0)

    def _get_pick_plan(self, obs: PyBulletState) -> list[PyBulletAction]:
        """Assume that the robot starts out empty-handed and near the books."""
        book_name = self._get_value("book")
        book_grasp = _book_grasp_to_pose(self._get_value("book_grasp"))
        return get_plan_to_pick_object(obs, book_name, book_grasp, self._sim)

    def _get_handover_plan(self, obs: PyBulletState) -> list[PyBulletAction]:
        """Assume that the robot starts out holding book and near person."""
        book_name = self._get_value("book")
        handover_pose = _handover_position_to_pose(self._get_value("handover_position"))
        return get_plan_to_handover_object(
            obs, book_name, handover_pose, self._sim, self._seed
        )


def _book_grasp_to_pose(yaw: NDArray) -> Pose:
    assert len(yaw) == 1
    return get_poses_facing_line(
        axis=(0.0, 0.0, 1.0),
        point_on_line=(0.0, 0.0, 0),
        radius=1e-3,
        num_points=1,
        angle_offset=yaw[0],
    )[0]


def _handover_position_to_pose(position: NDArray) -> Pose:
    handover_orientation = (
        0.8522037863731384,
        0.4745013415813446,
        -0.01094298530369997,
        0.22017613053321838,
    )
    return Pose(tuple(position), handover_orientation)


def _pose_is_reachable(pose: Pose, sim: PyBulletEnv) -> bool:
    try:
        inverse_kinematics(sim.robot, pose)
    except InverseKinematicsError:
        return False
    return True


def create_book_handover_csp(
    sim: PyBulletEnv, rom_model: ROMModel, preferred_books: list[str], seed: int = 0
) -> tuple[CSP, list[CSPSampler], CSPPolicy, dict[CSPVariable, Any]]:
    """Create a CSP for the task of handing over a book."""

    ################################ Variables ################################

    # Choose a book to fetch.
    book_names = [f"book{i}" for i in range(len(sim.book_ids))]
    assert set(preferred_books).issubset(book_names)
    book = CSPVariable("book", EnumSpace(book_names))

    # Choose a grasp on the book. Only the grasp yaw is unknown.
    book_grasp = CSPVariable("book_grasp", Box(-np.pi, np.pi, dtype=np.float_))

    # Choose a handover position. Relative to the resting hand position.
    handover_position = CSPVariable(
        "handover_position", Box(-np.inf, np.inf, shape=(3,), dtype=np.float_)
    )

    variables = [book, book_grasp, handover_position]

    ############################## Initialization #############################

    initialization = {
        book: book_names[0],
        book_grasp: np.zeros((1,)),
        handover_position: np.zeros((3,)),
    }

    ############################### Constraints ###############################

    # Create a user preference constraint for the book.
    def _book_is_preferred(book_name: str) -> bool:
        return book_name in preferred_books

    book_preference_constraint = CSPConstraint(
        "book_preference",
        [book],
        _book_is_preferred,
    )

    # Create a handover constraint given the user ROM.
    def _handover_position_is_in_rom(position: NDArray) -> bool:
        return rom_model.check_position_reachable(position)

    handover_rom_constraint = CSPConstraint(
        "handover_rom_constraint",
        [handover_position],
        _handover_position_is_in_rom,
    )

    # Create reaching constraints.
    def _book_grasp_is_reachable(yaw: NDArray) -> bool:
        pose = _book_grasp_to_pose(yaw)
        return _pose_is_reachable(pose, sim)

    book_grasp_reachable_constraint = CSPConstraint(
        "book_reachable",
        [book_grasp],
        _book_grasp_is_reachable,
    )

    def _handover_position_is_reachable(position: NDArray) -> bool:
        pose = _handover_position_to_pose(position)
        return _pose_is_reachable(pose, sim)

    handover_reachable_constraint = CSPConstraint(
        "handover_reachable",
        [handover_position],
        _handover_position_is_reachable,
    )

    constraints = [
        book_preference_constraint,
        handover_rom_constraint,
        book_grasp_reachable_constraint,
        handover_reachable_constraint,
    ]

    ################################### CSP ###################################

    csp = CSP(variables, constraints)

    ################################# Samplers ################################

    def _sample_book_fn(
        _: dict[CSPVariable, Any], rng: np.random.Generator
    ) -> dict[CSPVariable, Any]:
        preferred_book = preferred_books[rng.choice(len(preferred_books))]
        return {book: preferred_book}

    book_sampler = FunctionalCSPSampler(_sample_book_fn, csp, {book})

    def _sample_handover_pose(
        _: dict[CSPVariable, Any], rng: np.random.Generator
    ) -> dict[CSPVariable, Any]:
        position = rom_model.sample_reachable_position(rng)
        return {handover_position: position}

    handover_sampler = FunctionalCSPSampler(
        _sample_handover_pose, csp, {handover_position}
    )

    def _sample_grasp_pose(
        _: dict[CSPVariable, Any], rng: np.random.Generator
    ) -> dict[CSPVariable, Any]:
        yaw = rng.uniform(-np.pi, np.pi, size=(1,))
        return {book_grasp: yaw}

    grasp_sampler = FunctionalCSPSampler(_sample_grasp_pose, csp, {book_grasp})

    samplers: list[CSPSampler] = [book_sampler, handover_sampler, grasp_sampler]

    ################################# Policy ##################################

    policy: CSPPolicy = _BookHandoverCSPPolicy(sim, csp, seed=seed)

    return csp, samplers, policy, initialization
