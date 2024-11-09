"""CSP elements for the PyBullet environment."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
    sample_collision_free_inverse_kinematics,
)
from pybullet_helpers.math_utils import get_poses_facing_line
from tomsutils.llm import OpenAILLM
from tomsutils.spaces import EnumSpace

from multitask_personalization.csp_generation import CSPGenerator
from multitask_personalization.envs.pybullet.pybullet_env import (
    PyBulletEnv,
    get_user_book_enjoyment_logprob,
)
from multitask_personalization.envs.pybullet.pybullet_skills import (
    get_plan_to_handover_object,
    get_plan_to_pick_object,
    get_plan_to_place_object,
)
from multitask_personalization.envs.pybullet.pybullet_structs import (
    PyBulletAction,
    PyBulletState,
)
from multitask_personalization.rom.models import ROMModel, TrainableROMModel
from multitask_personalization.structs import (
    CSP,
    CSPConstraint,
    CSPCost,
    CSPPolicy,
    CSPSampler,
    CSPVariable,
    FunctionalCSPConstraint,
    FunctionalCSPSampler,
    LogProbCSPConstraint,
)
from multitask_personalization.utils import bernoulli_entropy


class _BookHandoverCSPPolicy(CSPPolicy[PyBulletState, PyBulletAction]):

    def __init__(
        self,
        sim: PyBulletEnv,
        csp: CSP,
        seed: int = 0,
        max_motion_planning_candidates: int = 1,
    ) -> None:
        super().__init__(csp, seed)
        self._sim = sim
        self._current_plan: list[PyBulletAction] = []
        self._max_motion_planning_candidates = max_motion_planning_candidates

    def reset(self, solution: dict[CSPVariable, Any]) -> None:
        super().reset(solution)
        self._current_plan = []

    def step(self, obs: PyBulletState) -> tuple[PyBulletAction, bool]:
        if not self._current_plan:
            if obs.held_object is None:
                self._current_plan = self._get_pick_plan(obs)
            elif obs.held_object == self._get_value("book"):
                self._current_plan = self._get_handover_plan(obs)
            else:
                self._current_plan = self._get_place_plan(obs)
        action = self._current_plan.pop(0)
        done = action[1] is None
        return action, done

    def _get_pick_plan(self, obs: PyBulletState) -> list[PyBulletAction]:
        """Assume that the robot starts out empty-handed and near the books."""
        self._sim.set_state(obs)
        book_description = self._get_value("book")
        book_grasp = _book_grasp_to_pose(self._get_value("book_grasp"))
        return get_plan_to_pick_object(
            obs,
            book_description,
            book_grasp,
            self._sim,
            max_motion_planning_candidates=self._max_motion_planning_candidates,
        )

    def _get_handover_plan(self, obs: PyBulletState) -> list[PyBulletAction]:
        """Assume that the robot starts out holding book and near person."""
        self._sim.set_state(obs)
        book_description = self._get_value("book")
        handover_pose = _handover_position_to_pose(self._get_value("handover_position"))
        handover_plan = get_plan_to_handover_object(
            obs,
            book_description,
            handover_pose,
            self._sim,
            self._seed,
            max_motion_planning_candidates=self._max_motion_planning_candidates,
        )
        # Finish the plan by indicating done.
        handover_plan.append((2, None))
        return handover_plan

    def _get_place_plan(self, obs: PyBulletState) -> list[PyBulletAction]:
        """The robot is holding the wrong thing; place it somewhere."""
        self._sim.set_state(obs)
        held_obj = obs.held_object
        assert held_obj is not None
        surface_name = "table"  # NOTE: placing back into the shelf not yet done
        surface_id = self._sim.get_object_id_from_name(surface_name)
        surface_extents = self._sim.get_aabb_dimensions(surface_id)
        held_obj_id = self._sim.get_object_id_from_name(held_obj)
        object_extents = self._sim.get_aabb_dimensions(held_obj_id)
        # In the near future, we will figure out how best to add placement pose
        # as another variable in the CSP, but only when the user is holding a
        # book that is not the one that was asked for. This is part of a bigger
        # thing which is figuring out how to make CSP generation conditional on
        # the current state (in a nice implementation kind of way).
        placement_lb = (
            -surface_extents[0] / 2 + object_extents[0] / 2,
            -surface_extents[1] / 2 + object_extents[1] / 2,
            surface_extents[2] / 2 + object_extents[2] / 2,
        )
        placement_ub = (
            surface_extents[0] / 2 - object_extents[0] / 2,
            surface_extents[1] / 2 - object_extents[1] / 2,
            surface_extents[2] / 2 + object_extents[2] / 2,
        )
        for _ in range(100000):
            placement_position = self._rng.uniform(placement_lb, placement_ub)
            relative_placement_pose = Pose.from_rpy(
                tuple(placement_position), (0, 0, np.pi / 2)
            )
            self._sim.set_state(obs)
            plan = get_plan_to_place_object(
                obs,
                held_obj,
                surface_name,
                relative_placement_pose,
                self._sim,
                max_motion_planning_time=1.0,
                max_motion_planning_candidates=self._max_motion_planning_candidates,
            )
            if plan is not None:
                return plan
        raise RuntimeError("Could not find placement plan")


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
        book_preference_initialization: str = "I like everything!",
        llm_model_name: str = "gpt-4o-mini",
        llm_cache_dir: Path = Path(__file__).parents[4] / "llm_cache",
        llm_max_tokens: int = 700,
        llm_use_cache_only: bool = False,
        llm_temperature: float = 1.0,
        max_motion_planning_candidates: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._sim = sim
        self._rom_model = rom_model
        self._rom_model_training_data: list[tuple[NDArray, bool]] = []
        self._current_book_preference = book_preference_initialization
        self._all_user_feedback: list[str] = []
        self._llm = OpenAILLM(
            llm_model_name,
            llm_cache_dir,
            max_tokens=llm_max_tokens,
            use_cache_only=llm_use_cache_only,
        )
        self._llm_temperature = llm_temperature
        self._max_motion_planning_candidates = max_motion_planning_candidates

    def _generate_variables(
        self,
        obs: PyBulletState,
    ) -> tuple[list[CSPVariable], dict[CSPVariable, Any]]:
        # Sync the simulator.
        self._sim.set_state(obs)

        # Choose a book to fetch.
        book = CSPVariable("book", EnumSpace(self._sim.book_descriptions))

        # Choose a grasp on the book. Only the grasp yaw is unknown.
        book_grasp = CSPVariable("book_grasp", Box(-np.pi, np.pi, dtype=np.float_))

        # Choose a handover position. Relative to the resting hand position.
        handover_position = CSPVariable(
            "handover_position", Box(-np.inf, np.inf, shape=(3,), dtype=np.float_)
        )

        variables = [book, book_grasp, handover_position]

        books = self._sim.book_descriptions
        init_book = books[self._rng.choice(len(books))]
        initialization = {
            book: init_book,
            book_grasp: np.array([-np.pi / 2]),
            handover_position: np.zeros((3,)),
        }

        return variables, initialization

    def _generate_personal_constraints(
        self,
        obs: PyBulletState,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:

        book, _, handover_position = variables

        book_preference_constraint = LogProbCSPConstraint(
            "book_preference",
            [book],
            self._book_is_preferred_logprob,
            threshold=np.log(0.5) - 1e-3,
        )

        # Create a handover constraint given the user ROM.
        def _handover_position_is_in_rom_logprob(position: NDArray) -> float:
            return self._rom_model.get_position_reachable_logprob(position)

        handover_rom_constraint = LogProbCSPConstraint(
            "handover_rom_constraint",
            [handover_position],
            _handover_position_is_in_rom_logprob,
        )

        return [book_preference_constraint, handover_rom_constraint]

    def _generate_nonpersonal_constraints(
        self,
        obs: PyBulletState,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:

        book, book_grasp, handover_position = variables

        # Create reaching constraints.
        def _book_grasp_is_reachable(yaw: NDArray) -> bool:
            pose = _book_grasp_to_pose(yaw)
            return _pose_is_reachable(pose, self._sim)

        book_grasp_reachable_constraint = FunctionalCSPConstraint(
            "book_reachable",
            [book_grasp],
            _book_grasp_is_reachable,
        )

        def _handover_position_is_reachable(position: NDArray) -> bool:
            pose = _handover_position_to_pose(position)
            handover_reachable = _pose_is_reachable(pose, self._sim)
            return handover_reachable

        handover_reachable_constraint = FunctionalCSPConstraint(
            "handover_reachable",
            [handover_position],
            _handover_position_is_reachable,
        )

        # Create collision constraints.
        def _handover_position_is_collision_free(
            position: NDArray, book_description: str, yaw: NDArray
        ) -> bool:
            book_id = self._sim.get_object_id_from_name(book_description)
            end_effector_pose = _handover_position_to_pose(position)
            grasp_pose = _book_grasp_to_pose(yaw)
            collision_bodies = self._sim.get_collision_ids() - {book_id}
            samples = list(
                sample_collision_free_inverse_kinematics(
                    self._sim.robot,
                    end_effector_pose,
                    collision_bodies,
                    self._rng,
                    held_object=book_id,
                    base_link_to_held_obj=grasp_pose.invert(),
                    max_candidates=1,
                )
            )
            assert len(samples) <= 1
            return len(samples) == 1

        handover_collision_free_constraint = FunctionalCSPConstraint(
            "handover_collision_free",
            [handover_position, book, book_grasp],
            _handover_position_is_collision_free,
        )

        return [
            book_grasp_reachable_constraint,
            handover_reachable_constraint,
            handover_collision_free_constraint,
        ]

    def _generate_exploit_cost(
        self,
        obs: PyBulletState,
        variables: list[CSPVariable],
    ) -> CSPCost | None:
        return None

    def _generate_samplers(
        self,
        obs: PyBulletState,
        csp: CSP,
    ) -> list[CSPSampler]:

        book, book_grasp, handover_position = csp.variables

        def _sample_book_fn(
            _: dict[CSPVariable, Any], rng: np.random.Generator
        ) -> dict[CSPVariable, Any]:
            book_description = self._sim.book_descriptions[
                rng.choice(len(self._sim.book_descriptions))
            ]
            return {book: book_description}

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

        return [book_sampler, handover_sampler, grasp_sampler]

    def _generate_policy(
        self,
        obs: PyBulletState,
        csp: CSP,
    ) -> CSPPolicy:
        return _BookHandoverCSPPolicy(
            self._sim,
            csp,
            seed=self._seed,
            max_motion_planning_candidates=self._max_motion_planning_candidates,
        )

    def learn_from_transition(
        self,
        obs: PyBulletState,
        act: PyBulletAction,
        next_obs: PyBulletState,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        self._update_rom_model(obs, act, next_obs, reward)
        self._update_book_preferences(act, next_obs, reward)

    def _book_is_preferred_logprob(self, book_description: str) -> float:
        return get_user_book_enjoyment_logprob(
            book_description, self._current_book_preference, self._llm, seed=self._seed
        )

    def _update_rom_model(
        self,
        obs: PyBulletState,
        act: PyBulletAction,
        next_obs: PyBulletState,
        reward: float,
    ) -> None:
        # Only train trainable ROM models.
        if not isinstance(self._rom_model, TrainableROMModel):
            return
        # Only learn from cases where the robot triggered "done".
        if not np.isclose(act[0], 2):
            return
        # If the user said something and the reward was negative, the failure
        # was due to book preferences, not handover failure.
        if reward < 0 and next_obs.human_text is not None:
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

    def _update_book_preferences(
        self, act: PyBulletAction, next_obs: PyBulletState, reward: float
    ) -> None:
        # Only learn when the user had something to say.
        if next_obs.human_text is None:
            return
        # For now, only learn when the robot triggered "done".
        if not np.isclose(act[0], 2):
            return
        assert act[1] is None
        assert next_obs.held_object is not None
        # Update the history of things the user has told the robot.
        result = "accepted" if reward > 0 else "rejected"
        new_feedback = f'When I gave the user the book: "{next_obs.held_object}", they {result} it and said: "{next_obs.human_text}"'  # pylint: disable=line-too-long
        self._all_user_feedback.append(new_feedback)
        # Learn from the history of all feedback.
        # For now, just do this once; in the future, get a distribution of
        # possibilities.
        all_feedback_str = "\n".join(self._all_user_feedback)
        # pylint: disable=line-too-long
        prompt = f"""Below is a first-person history of interactions between you, a robot, and a single human user:

{all_feedback_str}

Based on this history, concisely describe the user's taste in books.

NOTE: you should list an example or two of books that the user loves and another example or two of books that the user hates.

Return this description and nothing else. Do not explain anything."""
        response = self._llm.sample_completions(
            prompt,
            imgs=None,
            temperature=self._llm_temperature,
            seed=self._seed,
        )[0]
        self._current_book_preference = response
        logging.info(f"Updated learned user book preferences: {response}")

    def get_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if isinstance(self._rom_model, TrainableROMModel):
            metrics.update(self._rom_model.get_metrics())
        for book_description in self._sim.book_descriptions:
            lp = self._book_is_preferred_logprob(book_description)
            entropy = bernoulli_entropy(lp)
            metrics[f"entropy-{book_description}"] = entropy
        return metrics
