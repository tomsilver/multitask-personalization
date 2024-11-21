"""CSP elements for the PyBullet environment."""

import abc
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.spaces import Box, Discrete, Tuple
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
    sample_collision_free_inverse_kinematics,
)
from pybullet_helpers.manipulation import generate_surface_placements
from pybullet_helpers.math_utils import get_poses_facing_line
from pybullet_helpers.spaces import PoseSpace
from tomsutils.llm import LargeLanguageModel
from tomsutils.spaces import EnumSpace

from multitask_personalization.csp_generation import CSPGenerator
from multitask_personalization.envs.pybullet.pybullet_env import (
    PyBulletEnv,
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
from multitask_personalization.envs.pybullet.pybullet_utils import (
    get_user_book_enjoyment_logprob,
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


class _PyBulletCSPPolicy(CSPPolicy[PyBulletState, PyBulletAction]):

    def __init__(
        self,
        sim: PyBulletEnv,
        csp: CSP,
        seed: int = 0,
        max_motion_planning_time: float = np.inf,
        max_motion_planning_candidates: int = 1,
    ) -> None:
        super().__init__(csp, seed)
        self._sim = sim
        self._current_plan: list[PyBulletAction] = []
        self._max_motion_planning_time = max_motion_planning_time
        self._max_motion_planning_candidates = max_motion_planning_candidates
        self._terminated = False

    @abc.abstractmethod
    def _get_plan(self, obs: PyBulletState) -> list[PyBulletAction] | None:
        """Run planning."""

    @abc.abstractmethod
    def _policy_can_handle_mission(self, mission: str) -> bool:
        """Determine if this policy can do this mission."""

    def reset(self, solution: dict[CSPVariable, Any]) -> None:
        super().reset(solution)
        self._current_plan = []
        self._terminated = False

    def step(self, obs: PyBulletState) -> PyBulletAction:
        if not self._current_plan:
            self._sim.set_state(obs)
            plan = self._get_plan(obs)
            assert plan is not None
            self._current_plan = plan
        action = self._current_plan.pop(0)
        self._terminated = action[1] is None
        return action

    def check_termination(self, obs: PyBulletState) -> bool:
        if self._terminated:
            return True
        mission = _infer_mission_from_obs(obs)
        if mission is None:
            return False
        return not self._policy_can_handle_mission(mission)


class _BookHandoverCSPPolicy(_PyBulletCSPPolicy):

    def _get_plan(self, obs: PyBulletState) -> list[PyBulletAction] | None:
        book_description = self._get_value("book")
        book_grasp = _book_grasp_to_pose(self._get_value("book_grasp"))
        handover_pose = _handover_position_to_pose(self._get_value("handover_position"))
        if obs.held_object is None:
            # Pick up the target book.
            return get_plan_to_pick_object(
                obs,
                book_description,
                book_grasp,
                self._sim,
                max_motion_planning_candidates=self._max_motion_planning_candidates,
            )
        if obs.held_object == book_description:
            # Handover the book.
            plan = get_plan_to_handover_object(
                obs,
                book_description,
                handover_pose,
                self._sim,
                self._seed,
                max_motion_planning_candidates=self._max_motion_planning_candidates,
            )
            plan.append((2, None))
            return plan
        raise NotImplementedError

    def _policy_can_handle_mission(self, mission: str) -> bool:
        return mission == "hand over book"


class _PutAwayHeldObjectCSPPolicy(_PyBulletCSPPolicy):

    def _get_plan(self, obs: PyBulletState) -> list[PyBulletAction] | None:
        placement_pose = self._get_value("placement")
        surface_name, surface_link_id = self._get_value("surface")
        assert obs.held_object is not None
        return get_plan_to_place_object(
            obs,
            obs.held_object,
            surface_name,
            placement_pose,
            self._sim,
            max_motion_planning_time=self._max_motion_planning_time,
            max_motion_planning_candidates=self._max_motion_planning_candidates,
            surface_link_id=surface_link_id,
        )

    def _policy_can_handle_mission(self, mission: str) -> bool:
        return mission == "put away held object"


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


def _infer_mission_from_obs(obs: PyBulletState) -> str | None:
    # Hardcode rules to save some LLM costs for now.
    if obs.human_text is None:
        return None
    if "Please bring me a book to read" in obs.human_text:
        return "hand over book"
    if "Put away the thing you're holding" in obs.human_text:
        return "put away held object"
    return None


class PyBulletCSPGenerator(CSPGenerator[PyBulletState, PyBulletAction]):
    """Generate CSPs for the pybullet environment."""

    def __init__(
        self,
        sim: PyBulletEnv,
        rom_model: ROMModel,
        llm: LargeLanguageModel,
        book_preference_initialization: str = "Unknown",
        max_motion_planning_candidates: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._sim = sim
        self._rom_model = rom_model
        self._rom_model_training_data: list[tuple[NDArray, bool]] = []
        self._current_book_preference = book_preference_initialization
        self._all_user_feedback: list[str] = []
        self._llm = llm
        self._max_motion_planning_candidates = max_motion_planning_candidates
        self._current_mission: str | None = None

    def generate(
        self,
        obs: PyBulletState,
    ) -> tuple[
        CSP,
        list[CSPSampler],
        CSPPolicy[PyBulletState, PyBulletAction],
        dict[CSPVariable, Any],
    ]:
        # Important for the initial time step.
        self._update_current_mission(obs)
        return super().generate(obs)

    def save(self, model_dir: Path) -> None:
        # Save ROM model.
        self._rom_model.save(model_dir)
        # Save book preferences.
        book_preference_outfile = model_dir / "learned_book_preferences.txt"
        with open(book_preference_outfile, "w", encoding="utf-8") as f:
            f.write(self._current_book_preference)
        # Save all user feedback.
        user_feedback_dict = dict(enumerate(self._all_user_feedback))
        user_feedback_file = model_dir / "user_feedback_history.json"
        with open(user_feedback_file, "w", encoding="utf-8") as f:
            json.dump(user_feedback_dict, f)

    def load(self, model_dir: Path) -> None:
        # Load ROM model.
        self._rom_model.load(model_dir)
        # Load book preferences.
        book_preference_outfile = model_dir / "learned_book_preferences.txt"
        with open(book_preference_outfile, "r", encoding="utf-8") as f:
            self._current_book_preference = f.read()
        # Load all user feedback.
        user_feedback_file = model_dir / "user_feedback_history.json"
        with open(user_feedback_file, "r", encoding="utf-8") as f:
            user_feedback_dict = json.load(f)
        self._all_user_feedback = [
            user_feedback_dict[i] for i in range(len(user_feedback_dict))
        ]

    def _generate_variables(
        self,
        obs: PyBulletState,
    ) -> tuple[list[CSPVariable], dict[CSPVariable, Any]]:

        # Sync the simulator.
        self._sim.set_state(obs)

        # NOTE: need to figure out a way to make this more scalable...
        if self._current_mission == "hand over book":

            # Choose a book to fetch.
            books = self._sim.book_descriptions
            # NOTE: this is a temporary hack to save ourselves the trouble of
            # considering the case where the robot is currently holding some
            # book but decides to handover another book, which would require
            # placing the currently held book first, and would therefore require
            # a larger CSP that is essentially a combination of this hand over
            # book CSP and the placement CSP below. We generally need to figure
            # out a more elegant way to compose/scale CSP generation.
            if obs.held_object is not None:
                assert obs.held_object in books
                books = [obs.held_object]  # only consider this one!
            book = CSPVariable("book", EnumSpace(books))

            # Choose a grasp on the book. Only the grasp yaw is unknown.
            book_grasp = CSPVariable("book_grasp", Box(-np.pi, np.pi, dtype=np.float_))

            # Choose a handover position. Relative to the resting hand position.
            handover_position = CSPVariable(
                "handover_position", Box(-np.inf, np.inf, shape=(3,), dtype=np.float_)
            )

            variables = [book, book_grasp, handover_position]

            init_book = books[self._rng.choice(len(books))]
            initialization = {
                book: init_book,
                book_grasp: np.array([-np.pi / 2]),
                handover_position: np.zeros((3,)),
            }

        elif self._current_mission == "put away held object":

            # Choose a placement pose.
            placement = CSPVariable("placement", PoseSpace())

            # Choose a surface (and link ID on the surface).
            surfaces = sorted(self._sim.get_surface_names())
            surface = CSPVariable(
                "surface", Tuple([EnumSpace(surfaces), Discrete(1000000, start=-1)])
            )

            variables = [placement, surface]

            init_surface = surfaces[self._rng.choice(len(surfaces))]
            init_surface_id = self._sim.get_object_id_from_name(init_surface)
            surface_link_ids = sorted(self._sim.get_surface_link_ids(init_surface_id))
            init_surface_link_id = surface_link_ids[
                self._rng.choice(len(surface_link_ids))
            ]
            initialization = {
                placement: Pose.identity(),
                surface: (init_surface, init_surface_link_id),
            }

        else:
            raise NotImplementedError

        return variables, initialization

    def _generate_personal_constraints(
        self,
        obs: PyBulletState,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:

        # NOTE: need to figure out a way to make this more scalable...
        if self._current_mission == "hand over book":
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
                threshold=np.log(0.5) - 1e-3,
            )
            return [book_preference_constraint, handover_rom_constraint]

        if self._current_mission == "put away held object":
            # Nothing personal about putting away an object.
            return []

        raise NotImplementedError

    def _generate_nonpersonal_constraints(
        self,
        obs: PyBulletState,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:

        # NOTE: need to figure out a way to make this more scalable...
        if self._current_mission == "hand over book":
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

        if self._current_mission == "put away held object":

            # Lazy (of me) and slow, but correct...
            placement, surface = variables

            def _plan_to_place_exists(
                placement_pose: Pose,
                surface_name_and_link: tuple[str, int],
            ) -> bool:
                assert obs.held_object is not None
                surface_name, surface_link_id = surface_name_and_link
                max_mp_candidates = self._max_motion_planning_candidates
                try:
                    plan = get_plan_to_place_object(
                        obs,
                        obs.held_object,
                        surface_name,
                        placement_pose,
                        self._sim,
                        max_motion_planning_time=1e-1,
                        max_motion_planning_candidates=max_mp_candidates,
                        surface_link_id=surface_link_id,
                    )
                except:  # pylint: disable=bare-except
                    return False
                return plan is not None

            plan_to_place_exists = FunctionalCSPConstraint(
                "plan_to_place_exists",
                [placement, surface],
                _plan_to_place_exists,
            )

            return [plan_to_place_exists]

        raise NotImplementedError

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

        # NOTE: need to figure out a way to make this more scalable...
        if self._current_mission == "hand over book":

            book, book_grasp, handover_position = csp.variables

            # See note above.
            books = self._sim.book_descriptions
            if obs.held_object is not None:
                assert obs.held_object in books
                books = [obs.held_object]  # only consider this one!

            def _sample_book_fn(
                _: dict[CSPVariable, Any], rng: np.random.Generator
            ) -> dict[CSPVariable, Any]:
                book_description = books[rng.choice(len(books))]
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

        if self._current_mission == "put away held object":
            placement, surface = csp.variables

            assert obs.held_object is not None
            held_obj_id = self._sim.get_object_id_from_name(obs.held_object)

            def _sample_placement_pose(
                sol: dict[CSPVariable, Any], rng: np.random.Generator
            ) -> dict[CSPVariable, Any]:
                surface_name, surface_link_id = sol[surface]
                surface_id = self._sim.get_object_id_from_name(surface_name)
                placement_pose = next(
                    generate_surface_placements(
                        held_obj_id,
                        surface_id,
                        rng,
                        self._sim.physics_client_id,
                        surface_link_id=surface_link_id,
                    )
                )
                return {placement: placement_pose}

            placement_sampler = FunctionalCSPSampler(
                _sample_placement_pose, csp, {placement}
            )

            surfaces = sorted(self._sim.get_surface_names())

            def _sample_surface(
                _: dict[CSPVariable, Any], rng: np.random.Generator
            ) -> dict[CSPVariable, Any]:
                surface_name = surfaces[rng.choice(len(surfaces))]
                surface_id = self._sim.get_object_id_from_name(surface_name)
                candidates = sorted(self._sim.get_surface_link_ids(surface_id))
                surface_link_id = candidates[rng.choice(len(candidates))]
                return {surface: (surface_name, surface_link_id)}

            surface_sampler = FunctionalCSPSampler(_sample_surface, csp, {surface})

            return [placement_sampler, surface_sampler]

        raise NotImplementedError

    def _generate_policy(
        self,
        obs: PyBulletState,
        csp: CSP,
    ) -> CSPPolicy:

        # NOTE: need to figure out a way to make this more scalable...
        if self._current_mission == "hand over book":

            return _BookHandoverCSPPolicy(
                self._sim,
                csp,
                seed=self._seed,
                max_motion_planning_candidates=self._max_motion_planning_candidates,
            )

        if self._current_mission == "put away held object":

            return _PutAwayHeldObjectCSPPolicy(
                self._sim,
                csp,
                seed=self._seed,
                max_motion_planning_candidates=self._max_motion_planning_candidates,
            )

        raise NotImplementedError

    def observe_transition(
        self,
        obs: PyBulletState,
        act: PyBulletAction,
        next_obs: PyBulletState,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        if not self._disable_learning:
            self._update_rom_model(obs, act, next_obs)
            self._update_book_preferences(act, next_obs)
        self._update_current_mission(obs)

    def _book_is_preferred_logprob(self, book_description: str) -> float:
        return get_user_book_enjoyment_logprob(
            book_description, self._current_book_preference, self._llm, seed=self._seed
        )

    def _update_rom_model(
        self,
        obs: PyBulletState,
        act: PyBulletAction,
        next_obs: PyBulletState,
    ) -> None:
        # Only train trainable ROM models.
        if not isinstance(self._rom_model, TrainableROMModel):
            return
        # Only learn from cases where the robot triggered "done".
        if not np.isclose(act[0], 2):
            return
        assert act[1] is None
        # Check if the trigger was successful.
        label = next_obs.human_text != "I can't reach there"
        # Get the current position.
        self._sim.set_state(obs)
        pose = self._sim.robot.forward_kinematics(obs.robot_joints)
        # Update the training data.
        self._rom_model_training_data.append((np.array(pose.position), label))
        # Retrain the ROM model.
        self._rom_model.train(self._rom_model_training_data)

    def _update_book_preferences(
        self,
        act: PyBulletAction,
        next_obs: PyBulletState,
    ) -> None:
        # Only learn when the user had something to say.
        if next_obs.human_text is None:
            return
        # Ignore failures due to ROM.
        if "I can't reach there" in next_obs.human_text:
            return
        # For now, only learn when the robot triggered "done".
        if not np.isclose(act[0], 2):
            return
        assert act[1] is None
        assert next_obs.held_object is not None
        # Update the history of things the user has told the robot.
        new_feedback = f'When I gave the user the book: "{next_obs.held_object}", they said: "{next_obs.human_text}"'  # pylint: disable=line-too-long
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
            temperature=1.0,
            seed=self._seed,
        )[0]
        self._current_book_preference = response
        logging.info(f"Updated learned user book preferences: {response}")

    def _update_current_mission(self, obs: PyBulletState) -> None:
        mission = _infer_mission_from_obs(obs)
        if mission is not None:
            self._current_mission = mission

    def get_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if isinstance(self._rom_model, TrainableROMModel):
            metrics.update(self._rom_model.get_metrics())
        for book_description in self._sim.book_descriptions:
            lp = self._book_is_preferred_logprob(book_description)
            entropy = bernoulli_entropy(lp)
            metrics[f"entropy-{book_description}"] = entropy
        return metrics
