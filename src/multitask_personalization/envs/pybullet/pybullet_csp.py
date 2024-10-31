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
from multitask_personalization.rom.models import ROMModel, TrainableROMModel
from multitask_personalization.structs import (
    CSP,
    CSPConstraint,
    CSPGenerator,
    CSPPolicy,
    CSPSampler,
    CSPVariable,
    FunctionalCSPSampler,
)


class _BookHandoverCSPPolicy(CSPPolicy[PyBulletState, PyBulletAction]):

    def __init__(
        self,
        sim: PyBulletEnv,
        csp: CSP,
        seed: int = 0,
    ) -> None:
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
        return get_plan_to_pick_object(
            obs,
            book_name,
            book_grasp,
            self._sim,
        )

    def _get_handover_plan(self, obs: PyBulletState) -> list[PyBulletAction]:
        """Assume that the robot starts out holding book and near person."""
        book_name = self._get_value("book")
        handover_pose = _handover_position_to_pose(self._get_value("handover_position"))
        handover_plan = get_plan_to_handover_object(
            obs,
            book_name,
            handover_pose,
            self._sim,
            self._seed,
        )
        # Finish the plan by indicating done.
        handover_plan.append((2, None))
        return handover_plan


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


class PyBulletCSPGenerator(CSPGenerator[PyBulletState, PyBulletAction]):
    """Generate CSPs for the pybullet environment."""

    def __init__(
        self,
        sim: PyBulletEnv,
        rom_model: ROMModel,
        preferred_books: list[str],
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed)
        self._sim = sim
        self._rom_model = rom_model
        self._preferred_books = preferred_books
        self._rom_model_training_data: list[tuple[NDArray, bool]] = []

    def generate(self, obs: PyBulletState, explore: bool = False) -> tuple[
        CSP,
        list[CSPSampler],
        CSPPolicy[PyBulletState, PyBulletAction],
        dict[CSPVariable, Any],
    ]:

        self._sim.set_state(obs)

        # Create a CSP for the task of handing over a book.

        ################################ Variables ################################

        # Choose a book to fetch.
        book_names = [f"book{i}" for i in range(len(self._sim.book_ids))]
        assert set(self._preferred_books).issubset(book_names)
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

        constraints: list[CSPConstraint] = []

        if not explore:
            # Create a user preference constraint for the book.
            def _book_is_preferred(book_name: str) -> bool:
                book_preferred = book_name in self._preferred_books
                print("book_preferred:", book_preferred)
                return book_preferred

            book_preference_constraint = CSPConstraint(
                "book_preference",
                [book],
                _book_is_preferred,
            )
            constraints.append(book_preference_constraint)

            # Create a handover constraint given the user ROM.
            def _handover_position_is_in_rom(position: NDArray) -> bool:
                handover_in_rom = self._rom_model.check_position_reachable(position)
                print("handover_in_rom:", handover_in_rom)
                return handover_in_rom

            handover_rom_constraint = CSPConstraint(
                "handover_rom_constraint",
                [handover_position],
                _handover_position_is_in_rom,
            )
            constraints.append(handover_rom_constraint)

        # Create reaching constraints.
        def _book_grasp_is_reachable(yaw: NDArray) -> bool:
            pose = _book_grasp_to_pose(yaw)
            grasp_reachable = _pose_is_reachable(pose, self._sim)
            print("grasp_reachable:", grasp_reachable)
            return grasp_reachable

        book_grasp_reachable_constraint = CSPConstraint(
            "book_reachable",
            [book_grasp],
            _book_grasp_is_reachable,
        )
        constraints.append(book_grasp_reachable_constraint)

        def _handover_position_is_reachable(position: NDArray) -> bool:
            pose = _handover_position_to_pose(position)
            handover_reachable = _pose_is_reachable(pose, self._sim)
            print("handover_reachable", handover_reachable)
            return handover_reachable

        handover_reachable_constraint = CSPConstraint(
            "handover_reachable",
            [handover_position],
            _handover_position_is_reachable,
        )
        constraints.append(handover_reachable_constraint)

        ################################### CSP ###################################

        csp = CSP(variables, constraints)

        ################################# Samplers ################################

        def _sample_book_fn(
            _: dict[CSPVariable, Any], rng: np.random.Generator
        ) -> dict[CSPVariable, Any]:
            preferred_book = self._preferred_books[
                rng.choice(len(self._preferred_books))
            ]
            return {book: preferred_book}

        book_sampler = FunctionalCSPSampler(_sample_book_fn, csp, {book})

        def _sample_handover_pose(
            _: dict[CSPVariable, Any], rng: np.random.Generator
        ) -> dict[CSPVariable, Any]:
            position = self._rom_model.sample_reachable_position(rng)
            return {handover_position: position}

        handover_sampler = FunctionalCSPSampler(
            _sample_handover_pose, csp, {handover_position}
        )

        def _sample_grasp_pose(
            _: dict[CSPVariable, Any], rng: np.random.Generator
        ) -> dict[CSPVariable, Any]:
            del rng  # not actually sampling right now, for simplicity
            yaw = np.array([-np.pi / 2])
            return {book_grasp: yaw}

        grasp_sampler = FunctionalCSPSampler(_sample_grasp_pose, csp, {book_grasp})

        samplers: list[CSPSampler] = [book_sampler, handover_sampler, grasp_sampler]

        ################################# Policy ##################################

        policy: CSPPolicy = _BookHandoverCSPPolicy(
            self._sim,
            csp,
            seed=self._seed,
        )

        return csp, samplers, policy, initialization

    def learn_from_transition(
        self,
        obs: PyBulletState,
        act: PyBulletAction,
        next_obs: PyBulletState,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        # Only train trainable ROM models.
        if not isinstance(self._rom_model, TrainableROMModel):
            return
        # Only learn from cases where the robot triggered "done".
        if not np.isclose(act[0], 2):
            return
        assert act[1] is None
        # Check if the trigger was successful.
        label = reward > 0
        # Get the current position.
        self._sim.set_state(obs)
        pose = self._sim.robot.forward_kinematics(obs.robot_joints)
        # Update the training data.
        self._rom_model_training_data.append((np.array(pose.position), label))
        # Retrain the ROM model.
        self._rom_model.train(self._rom_model_training_data)

    def get_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if isinstance(self._rom_model, TrainableROMModel):
            metrics.update(self._rom_model.get_metrics())
        return metrics
