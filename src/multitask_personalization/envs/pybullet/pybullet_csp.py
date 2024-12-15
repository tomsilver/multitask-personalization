"""CSP elements for the PyBullet environment."""

import abc
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.spaces import Box, Discrete, Tuple
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, multiply_poses, set_pose, get_pose
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
    sample_collision_free_inverse_kinematics,
)
from pybullet_helpers.link import get_link_pose
from pybullet_helpers.manipulation import generate_surface_placements
from pybullet_helpers.math_utils import get_poses_facing_line
from pybullet_helpers.spaces import PoseSpace
from tomsutils.llm import LargeLanguageModel
from tomsutils.spaces import EnumSpace
from tomsutils.utils import create_rng_from_rng

from multitask_personalization.csp_generation import CSPGenerator
from multitask_personalization.envs.pybullet.pybullet_env import (
    PyBulletEnv,
)
from multitask_personalization.envs.pybullet.pybullet_skills import (
    get_duster_head_frame_wiping_plan,
    get_plan_to_handover_object,
    get_plan_to_move_to_pose,
    get_plan_to_pick_object,
    get_plan_to_place_object,
    get_plan_to_retract,
    get_plan_to_wipe_surface,
    get_target_base_pose,
    get_plan_to_reverse_handover_object,
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
        self._terminated = bool(np.isclose(action[0], 2) and action[1] == "Done")
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
        book_grasp = _book_grasp_to_relative_pose(self._get_value("book_grasp"))
        handover_pose = _handover_position_to_pose(self._get_value("handover_position"))
        grasp_base_pose = self._get_value("grasp_base_pose")
        assert isinstance(grasp_base_pose, Pose)
        handover_base_pose = self._get_value("handover_base_pose")
        assert isinstance(handover_base_pose, Pose)

        # Retract after transfer.
        if obs.human_held_object is not None:
            assert obs.human_held_object == book_description
            plan = get_plan_to_retract(
                obs,
                self._sim,
                obs.human_held_object,
                max_motion_planning_time=self._max_motion_planning_time,
            )
            # Indicate done.
            plan.append((2, "Done"))
            return plan
        if obs.held_object is None:
            # First move next to the object.
            if not grasp_base_pose.allclose(obs.robot_base, atol=1e-3):
                return get_plan_to_move_to_pose(
                    obs, grasp_base_pose, self._sim, seed=self._seed
                )
            # Pick up the target book.
            return get_plan_to_pick_object(
                obs,
                book_description,
                book_grasp,
                self._sim,
                max_motion_planning_candidates=self._max_motion_planning_candidates,
            )
        if obs.held_object == book_description:
            # If the book is already ready for handover, we are waiting for the
            # human to grasp it.
            self._sim.set_robot_base(obs.robot_base)
            ee_pose = self._sim.robot.forward_kinematics(obs.robot_joints)
            if ee_pose.allclose(handover_pose, atol=1e-3):
                return [(3, None)]  # waiting
            # Move to the handover base pose.
            if not handover_base_pose.allclose(obs.robot_base, atol=1e-3):
                return get_plan_to_move_to_pose(
                    obs, handover_base_pose, self._sim, seed=self._seed
                )
            # Handover the book.
            plan = get_plan_to_handover_object(
                obs,
                book_description,
                handover_pose,
                self._sim,
                self._seed,
                max_motion_planning_candidates=self._max_motion_planning_candidates,
            )
            # Tell the human to take the book.
            plan.append((2, "Here you go!"))
            return plan
        # Need to place held object.
        placement_pose = self._get_value("placement")
        placement_base_pose = self._get_value("placement_base_pose")
        assert isinstance(placement_base_pose, Pose)
        # Move to the placement base pose.
        if not placement_base_pose.allclose(obs.robot_base, atol=1e-3):
            return get_plan_to_move_to_pose(
                obs, placement_base_pose, self._sim, seed=self._seed
            )
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
        return mission == "hand over book"


class _PutAwayRobotHeldObjectCSPPolicy(_PyBulletCSPPolicy):

    def _get_plan(self, obs: PyBulletState) -> list[PyBulletAction] | None:
        placement_pose = self._get_value("placement")
        surface_name, surface_link_id = self._get_value("surface")
        placement_base_pose = self._get_value("placement_base_pose")
        assert isinstance(placement_base_pose, Pose)
        # Move to the placement base pose.
        if not placement_base_pose.allclose(obs.robot_base, atol=1e-3):
            return get_plan_to_move_to_pose(
                obs, placement_base_pose, self._sim, seed=self._seed
            )
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
        return mission == "put away robot held object"


class _PutAwayHumanHeldObjectCSPPolicy(_PyBulletCSPPolicy):

    def _get_plan(self, obs: PyBulletState) -> list[PyBulletAction] | None:
        # Put away the main object.
        if obs.held_object is not None and obs.human_held_object is None:
            placement_pose = self._get_value("placement")
            surface_name, surface_link_id = self._get_value("surface")
            placement_base_pose = self._get_value("placement_base_pose")
            assert isinstance(placement_base_pose, Pose)
            # Move to the placement base pose.
            if not placement_base_pose.allclose(obs.robot_base, atol=1e-3):
                return get_plan_to_move_to_pose(
                    obs, placement_base_pose, self._sim, seed=self._seed
                )
            place_plan = get_plan_to_place_object(
                obs,
                obs.held_object,
                surface_name,
                placement_pose,
                self._sim,
                max_motion_planning_time=self._max_motion_planning_time,
                max_motion_planning_candidates=self._max_motion_planning_candidates,
                surface_link_id=surface_link_id,
            )
            assert place_plan is not None
            return place_plan
        # Put away the object that we're holding at first.
        if obs.held_object is not None:
            placement_pose = self._get_value("first_placement")
            surface_name, surface_link_id = self._get_value("first_surface")
            placement_base_pose = self._get_value("first_placement_base_pose")
            assert isinstance(placement_base_pose, Pose)
            # Move to the placement base pose.
            if not placement_base_pose.allclose(obs.robot_base, atol=1e-3):
                return get_plan_to_move_to_pose(
                    obs, placement_base_pose, self._sim, seed=self._seed
                )
            place_plan = get_plan_to_place_object(
                obs,
                obs.held_object,
                surface_name,
                placement_pose,
                self._sim,
                max_motion_planning_time=self._max_motion_planning_time,
                max_motion_planning_candidates=self._max_motion_planning_candidates,
                surface_link_id=surface_link_id,
            )
            assert place_plan is not None
            return place_plan
        # Reverse handover.
        grasp_base_pose = self._get_value("grasp_base_pose")
        # First move next to the object.
        if not grasp_base_pose.allclose(obs.robot_base, atol=1e-3):
            return get_plan_to_move_to_pose(
                obs, grasp_base_pose, self._sim, seed=self._seed
            )
        # Do the reverse handover.
        grasp_yaw = np.array([-np.pi / 2])
        relative_grasp = _book_grasp_to_relative_pose(grasp_yaw)

        return get_plan_to_reverse_handover_object(
            obs,
            obs.human_held_object,
            relative_grasp,
            self._sim,
            max_motion_planning_candidates=self._max_motion_planning_candidates,
            max_motion_planning_time=self._max_motion_planning_time,
        )

    def _policy_can_handle_mission(self, mission: str) -> bool:
        return mission == "put away human held object"


class _CleanCSPPolicy(_PyBulletCSPPolicy):

    def _get_plan(self, obs: PyBulletState) -> list[PyBulletAction] | None:
        surface_name, link_id = self._get_value("surface")
        base_pose, joint_arr = self._get_value("robot_state")
        grasp_base_pose = self._get_value("grasp_base_pose")
        assert isinstance(grasp_base_pose, Pose)
        joint_state = joint_arr.tolist()
        num_rots = 1 if surface_name == "table" else 0
        if obs.held_object is None or obs.held_object == "duster":
            plan = get_plan_to_wipe_surface(
                obs,
                "duster",
                surface_name,
                grasp_base_pose,
                base_pose,
                joint_state,
                num_rots,
                self._sim,
                surface_link_id=link_id,
            )
            assert plan is not None
            # Indicate done.
            plan.append((2, "Done"))
            return plan
        # Need to place held object.
        placement_pose = self._get_value("placement")
        surface_name, surface_link_id = self._get_value("placement_surface")
        assert obs.held_object is not None
        placement_base_pose = self._get_value("placement_base_pose")
        assert isinstance(placement_base_pose, Pose)
        # Move to the placement base pose.
        if not placement_base_pose.allclose(obs.robot_base, atol=1e-3):
            return get_plan_to_move_to_pose(
                obs, placement_base_pose, self._sim, seed=self._seed
            )
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
        return mission == "clean"

    def check_termination(self, obs: PyBulletState) -> bool:
        if obs.human_text is not None and "Don't clean" in obs.human_text:
            return True
        return super().check_termination(obs)


def _book_grasp_to_relative_pose(yaw: NDArray) -> Pose:
    assert len(yaw) == 1
    return get_poses_facing_line(
        axis=(0.0, 0.0, 1.0),
        point_on_line=(0.0, 0.0, 0),
        radius=1e-3,
        num_points=1,
        angle_offset=yaw[0],
    )[0]


def _handover_position_to_pose(position: NDArray) -> Pose:
    handover_rpy = (-np.pi / 4, np.pi / 2, 0.0)
    return Pose.from_rpy(tuple(position), handover_rpy)


def _pose_is_reachable(pose: Pose, robot_base_pose, sim: PyBulletEnv) -> bool:
    sim.set_robot_base(robot_base_pose)
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
        return "put away robot held object"
    if "Put this away" in obs.human_text:
        return "put away human held object"
    if "Clean the dirty surfaces" in obs.human_text:
        return "clean"
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
        self._surface_can_be_cleaned: dict[tuple[str, int], bool | str] = {
            (surface, l): "unknown"
            for surface in sim.get_surface_names()
            for l in sim.get_surface_link_ids(sim.get_object_id_from_name(surface))
        }
        self._steps_since_last_cleaning_admonishment: int | None = None
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
            user_feedback_dict[str(i)] for i in range(len(user_feedback_dict))
        ]

    def _generate_variables(
        self,
        obs: PyBulletState,
    ) -> tuple[list[CSPVariable], dict[CSPVariable, Any]]:

        logging.info(f"Generating CSP for: {self._current_mission}")

        # Sync the simulator.
        self._sim.set_state(obs)

        # NOTE: need to figure out a way to make this more scalable...
        if self._current_mission == "hand over book":

            # Choose a book to fetch.
            books = self._sim.book_descriptions
            book = CSPVariable("book", EnumSpace(books))

            # Choose a grasp on the book. Only the grasp yaw is unknown.
            book_grasp = CSPVariable("book_grasp", Box(-np.pi, np.pi, dtype=np.float_))

            # Choose a handover position. Relative to the resting hand position.
            handover_position = CSPVariable(
                "handover_position", Box(-np.inf, np.inf, shape=(3,), dtype=np.float_)
            )

            # Choose a base pose for grasping the book.
            grasp_base_pose = CSPVariable("grasp_base_pose", PoseSpace())

            # Choose a base pose for handing over the book.
            handover_base_pose = CSPVariable("handover_base_pose", PoseSpace())

            variables = [
                book,
                book_grasp,
                handover_position,
                grasp_base_pose,
                handover_base_pose,
            ]

            init_book = books[self._rng.choice(len(books))]
            if obs.held_object == init_book:
                init_grasp_base_pose = obs.robot_base
            else:
                init_surface = self._sim.get_name_from_object_id(
                    self._sim.get_surface_that_object_is_on(
                        self._sim.get_object_id_from_name(init_book)
                    )
                )
                init_grasp_base_pose = get_target_base_pose(
                    obs, init_surface, self._sim
                )
            initialization = {
                book: init_book,
                book_grasp: np.array([-np.pi / 2]),
                handover_position: np.zeros((3,)),
                grasp_base_pose: init_grasp_base_pose,
                handover_base_pose: get_target_base_pose(obs, "bed", self._sim),
            }

            if obs.held_object is not None:
                # If the user is holding something, we'll need to place it, and
                # we'll need to determine a placement for it as part of the CSP.
                placement_vars, placement_init = self._generate_placement_variables(
                    obs.robot_base
                )
                variables.extend(placement_vars)
                initialization.update(placement_init)

        elif self._current_mission == "put away robot held object":

            variables, initialization = self._generate_placement_variables(
                obs.robot_base
            )

        elif self._current_mission == "put away human held object":

            # Choose a base pose for grasping the human held object.
            grasp_base_pose = CSPVariable("grasp_base_pose", PoseSpace())

            # Choose a grasp on the held object. Only the grasp yaw is unknown.
            grasp = CSPVariable("grasp", Box(-np.pi, np.pi, dtype=np.float_))

            # Choose a base pose for placing the human held object.
            placement_variables, placement_initialization = (
                self._generate_placement_variables(obs.robot_base)
            )

            variables = [
                grasp_base_pose,
                grasp,
            ] + placement_variables

            init_grasp_base_pose = get_target_base_pose(obs, "bed", self._sim)
            init_grasp = np.array([-np.pi / 2])

            initialization = {
                grasp_base_pose: init_grasp_base_pose,
                grasp: init_grasp,
                **placement_initialization,
            }

            if obs.held_object is not None:
                # If the user is holding something, we'll need to place it, and
                # we'll need to determine a placement for it as part of the CSP.
                first_placement_vars, first_placement_init = (
                    self._generate_placement_variables(
                        obs.robot_base,
                        placement_name="first_placement",
                        surface_name="first_surface",
                        placement_base_pose_name="first_placement_base_pose",
                    )
                )
                variables.extend(first_placement_vars)
                initialization.update(first_placement_init)

        elif self._current_mission == "clean":

            # Choose a surface to clean and a robot base pose / joint state to
            # initiate the cleaning. Also determine where to stand while picking
            # the duster.
            surface, surface_init = self._generate_surface_variable()
            robot_state = CSPVariable(
                "robot_state",
                Tuple(
                    [
                        PoseSpace(),
                        Box(
                            np.array(self._sim.robot.joint_lower_limits),
                            np.array(self._sim.robot.joint_upper_limits),
                        ),
                    ]
                ),
            )
            grasp_base_pose = CSPVariable("grasp_base_pose", PoseSpace())

            variables = [surface, robot_state, grasp_base_pose]

            init_robot_base_pose = obs.robot_base
            init_robot_joint_state = np.array(obs.robot_joints)
            if obs.held_object == "duster":
                init_grasp_base_pose = obs.robot_base
            else:
                init_grasp_base_pose = get_target_base_pose(obs, "duster", self._sim)
            initialization = {
                surface: surface_init,
                robot_state: (init_robot_base_pose, init_robot_joint_state),
                grasp_base_pose: init_grasp_base_pose,
            }

            if obs.held_object is not None:
                # If the user is holding something, we'll need to place it, and
                # we'll need to determine a placement for it as part of the CSP.
                placement_vars, placement_init = self._generate_placement_variables(
                    default_base_pose=obs.robot_base, surface_name="placement_surface"
                )
                variables.extend(placement_vars)
                initialization.update(placement_init)

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
            book, _, handover_position = variables[:3]

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

        if self._current_mission == "put away robot held object":
            # Nothing personal about putting away an object.
            return []

        if self._current_mission == "put away human held object":
            return []

        if self._current_mission == "clean":
            # The user may prefer to clean certain surfaces themselves (e.g,.
            # for the sake of feeling empowered, or because they do it better).

            surface = variables[0]

            def _robot_can_clean_surface_logprob(surface: tuple[str, int]) -> float:
                status = self._surface_can_be_cleaned[surface]
                assert status in (True, False, "unknown")
                if status == "unknown":
                    return np.log(0.5)
                if status:
                    return 0.0
                return -np.inf

            surfaces_to_clean_constraint = LogProbCSPConstraint(
                "surfaces_to_clean_constraint",
                [surface],
                _robot_can_clean_surface_logprob,
                threshold=np.log(0.5) - 1e-3,
            )
            return [surfaces_to_clean_constraint]

        raise NotImplementedError

    def _generate_nonpersonal_constraints(
        self,
        obs: PyBulletState,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:

        # NOTE: need to figure out a way to make this more scalable...
        if self._current_mission == "hand over book":
            book, book_grasp, handover_position, grasp_base_pose, handover_base_pose = (
                variables[:5]
            )

            # Create reaching constraints.
            def _book_grasp_is_reachable(
                book_description: str, yaw: NDArray, base_pose: Pose
            ) -> bool:
                if obs.held_object == book_description:
                    return True
                relative_pose = _book_grasp_to_relative_pose(yaw)
                book_idx = obs.book_descriptions.index(book_description)
                book_pose = obs.book_poses[book_idx]
                world_pose = multiply_poses(book_pose, relative_pose)
                return _pose_is_reachable(world_pose, base_pose, self._sim)

            book_grasp_reachable_constraint = FunctionalCSPConstraint(
                "book_reachable",
                [book, book_grasp, grasp_base_pose],
                _book_grasp_is_reachable,
            )

            def _handover_position_is_reachable(
                position: NDArray, base_pose: Pose
            ) -> bool:
                pose = _handover_position_to_pose(position)
                handover_reachable = _pose_is_reachable(pose, base_pose, self._sim)
                return handover_reachable

            handover_reachable_constraint = FunctionalCSPConstraint(
                "handover_reachable",
                [handover_position, handover_base_pose],
                _handover_position_is_reachable,
            )

            # Create collision constraints.
            def _handover_position_is_collision_free(
                position: NDArray,
                book_description: str,
                yaw: NDArray,
                base_pose: Pose,
            ) -> bool:
                self._sim.set_robot_base(base_pose)
                book_id = self._sim.get_object_id_from_name(book_description)
                end_effector_pose = _handover_position_to_pose(position)
                grasp_pose = _book_grasp_to_relative_pose(yaw)
                collision_bodies = self._sim.get_collision_ids() - {book_id}
                if obs.held_object is not None:
                    collision_bodies -= {
                        self._sim.get_object_id_from_name(obs.held_object)
                    }
                # The number of calls to the RNG internally to the function is
                # nondeterministic, so make a new RNG to maintain determinism.
                ik_rng = create_rng_from_rng(self._rng)
                samples = list(
                    sample_collision_free_inverse_kinematics(
                        self._sim.robot,
                        end_effector_pose,
                        collision_bodies,
                        ik_rng,
                        held_object=book_id,
                        base_link_to_held_obj=grasp_pose.invert(),
                        max_candidates=1,
                    )
                )
                assert len(samples) <= 1
                return len(samples) == 1

            handover_collision_free_constraint = FunctionalCSPConstraint(
                "handover_collision_free",
                [handover_position, book, book_grasp, handover_base_pose],
                _handover_position_is_collision_free,
            )

            constraints: list[CSPConstraint] = [
                book_grasp_reachable_constraint,
                handover_reachable_constraint,
                handover_collision_free_constraint,
            ]

            if obs.held_object is not None:
                assert len(variables) == 8
                placement, surface, placement_base_pose = variables[5:]
                plan_to_place_exists = self._generate_plan_to_place_exists_constraint(
                    obs, placement, surface, placement_base_pose
                )
                constraints.append(plan_to_place_exists)

            return constraints

        if self._current_mission == "put away robot held object":
            placement, surface, placement_base_pose = variables
            plan_to_place_exists = self._generate_plan_to_place_exists_constraint(
                obs, placement, surface, placement_base_pose
            )
            return [plan_to_place_exists]

        if self._current_mission == "put away human held object":
            _, grasp_yaw, placement, surface, placement_base_pose = variables[:5]

            def _placement_after_grasp_exists(
                yaw: NDArray,
                placement_pose: Pose,
                surface_name_and_link: tuple[str, int],
                base_pose: Pose,
                held_object_relative_placement: Pose | None = None,
                held_object_placement_surface: tuple[str, int] | None = None,
            ) -> bool:

                self._sim.set_state(obs)
                if (
                    self._sim.current_held_object_id is not None
                ):
                    assert held_object_relative_placement is not None
                    assert held_object_placement_surface is not None
                    placement_surface_id = self._sim.get_object_id_from_name(
                        held_object_placement_surface[0]
                    )
                    placement_surface_link_id = held_object_placement_surface[1]
                    placement_surface_link_pose = get_link_pose(
                        placement_surface_id,
                        placement_surface_link_id,
                        self._sim.physics_client_id,
                    )
                    absolute_placement = multiply_poses(
                        placement_surface_link_pose, held_object_relative_placement
                    )
                    set_pose(
                        self._sim.current_held_object_id,
                        absolute_placement,
                        self._sim.physics_client_id,
                    )
                    self._sim.current_held_object_id = None
                    self._sim.current_grasp_transform = None

                # Snap object to grasp.
                self._sim.set_robot_base(base_pose)
                grasp_pose = _book_grasp_to_relative_pose(yaw)
                end_effector_pose = self._sim.robot.get_end_effector_pose()
                held_obj_pose = multiply_poses(end_effector_pose, grasp_pose.invert())
                assert obs.human_held_object is not None
                held_obj_id = self._sim.get_object_id_from_name(obs.human_held_object)
                set_pose(held_obj_id, held_obj_pose, self._sim.physics_client_id)
                self._sim.current_held_object_id = held_obj_id
                self._sim.current_grasp_transform = grasp_pose.invert()
                surface_name, surface_link_id = surface_name_and_link
                obs_after_base_move = self._sim.get_state()

                max_mp_candidates = self._max_motion_planning_candidates
                plan = get_plan_to_place_object(
                    obs_after_base_move,
                    obs.human_held_object,
                    surface_name,
                    placement_pose,
                    self._sim,
                    max_motion_planning_time=1e-1,
                    max_motion_planning_candidates=max_mp_candidates,
                    surface_link_id=surface_link_id,
                )
                result = plan is not None
                return result

            plan_to_place_exists_vars = [grasp_yaw, placement, surface, placement_base_pose]
            if obs.held_object is not None:
                first_placement, first_placement_surface = variables[5:7]
                plan_to_place_exists_vars.extend([first_placement, first_placement_surface])

            plan_to_place_after_pick_exists = FunctionalCSPConstraint(
                "place-after-pick",
                plan_to_place_exists_vars,
                _placement_after_grasp_exists,
            )

            constraints = [plan_to_place_after_pick_exists]

            if obs.held_object is not None:
                assert len(variables) == 8
                first_placement, first_placement_surface, first_placement_base_pose = (
                    variables[5:]
                )
                first_plan_to_place_exists = (
                    self._generate_plan_to_place_exists_constraint(
                        obs,
                        first_placement,
                        first_placement_surface,
                        first_placement_base_pose,
                    )
                )
                constraints.append(first_plan_to_place_exists)

            return constraints

        if self._current_mission == "clean":

            # Coming soon: removing objects to clean surfaces.

            surface, robot_state, grasp_base_pose = variables[:3]

            def _prewipe_pose_is_valid(
                surface_name_and_link: tuple[str, int],
                candidate_robot_state: tuple[Pose, NDArray],
            ) -> bool:
                # Necessary to escape from initialization.
                surface_name, surface_link_id = surface_name_and_link
                base_pose, robot_joint_arr = candidate_robot_state
                num_rots = 1 if surface_name == "table" else 0
                self._sim.set_robot_base(base_pose)
                self._sim.robot.set_joints(robot_joint_arr.tolist())
                current_pose = self._sim.robot.get_end_effector_pose()
                target_pose = self._get_prewipe_end_effector_pose(
                    surface_name, surface_link_id, num_rots
                )
                return target_pose.allclose(current_pose, atol=1e-3)

            prewipe_pose_is_valid = FunctionalCSPConstraint(
                "prewipe_pose_is_valid",
                [surface, robot_state],
                _prewipe_pose_is_valid,
            )

            def _wipe_plan_exists(
                surface_name_and_link: tuple[str, int],
                candidate_robot_state: tuple[Pose, NDArray],
                grasp_base_pose: Pose,
                held_object_relative_placement: Pose | None = None,
                held_object_placement_surface: tuple[str, int] | None = None,
            ) -> bool:
                surface_name, surface_link_id = surface_name_and_link
                base_pose, robot_joint_arr = candidate_robot_state
                joint_state = robot_joint_arr.tolist()
                num_rots = 1 if surface_name == "table" else 0

                self._sim.set_state(obs)
                if (
                    self._sim.current_held_object_id is not None
                    and self._sim.current_held_object_id != self._sim.duster_id
                ):
                    assert held_object_relative_placement is not None
                    assert held_object_placement_surface is not None
                    placement_surface_id = self._sim.get_object_id_from_name(
                        held_object_placement_surface[0]
                    )
                    placement_surface_link_id = held_object_placement_surface[1]
                    placement_surface_link_pose = get_link_pose(
                        placement_surface_id,
                        placement_surface_link_id,
                        self._sim.physics_client_id,
                    )
                    absolute_placement = multiply_poses(
                        placement_surface_link_pose, held_object_relative_placement
                    )
                    set_pose(
                        self._sim.current_held_object_id,
                        absolute_placement,
                        self._sim.physics_client_id,
                    )
                    self._sim.current_held_object_id = None
                    self._sim.current_grasp_transform = None
                pre_wipe_state = self._sim.get_state()
                wipe_plan = get_plan_to_wipe_surface(
                    pre_wipe_state,
                    "duster",
                    surface_name,
                    grasp_base_pose,
                    base_pose,
                    joint_state,
                    num_rots,
                    self._sim,
                    surface_link_id=surface_link_id,
                    max_motion_planning_time=1e-1,
                )
                return wipe_plan is not None

            wipe_plan_exists_vars = [surface, robot_state, grasp_base_pose]
            if obs.held_object is not None and obs.held_object != "duster":
                placement, placement_surface = variables[3:5]
                wipe_plan_exists_vars.extend([placement, placement_surface])
            wipe_plan_exists = FunctionalCSPConstraint(
                "wipe_plan_exists",
                wipe_plan_exists_vars,
                _wipe_plan_exists,
            )

            constraints = [
                prewipe_pose_is_valid,
                wipe_plan_exists,
            ]

            if obs.held_object is not None:
                # Shortcut: if already holding the duster, don't bother making
                # a real plan to place it, because the policy won't need to.
                if obs.held_object != "duster":
                    assert len(variables) == 6
                    placement, placement_surface, placement_base_pose = variables[3:]
                    plan_to_place_exists = (
                        self._generate_plan_to_place_exists_constraint(
                            obs, placement, placement_surface, placement_base_pose
                        )
                    )
                    constraints.append(plan_to_place_exists)

            return constraints

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

            book, book_grasp, handover_position, grasp_base_pose = csp.variables[:4]

            books = self._sim.book_descriptions

            def _sample_book_fn(
                _: dict[CSPVariable, Any], rng: np.random.Generator
            ) -> dict[CSPVariable, Any]:
                book_description = books[rng.choice(len(books))]
                if obs.held_object == book_description:
                    base_pose = obs.robot_base
                else:
                    base_pose = get_target_base_pose(obs, book_description, self._sim)
                return {book: book_description, grasp_base_pose: base_pose}

            book_sampler = FunctionalCSPSampler(
                _sample_book_fn, csp, {book, grasp_base_pose}
            )

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

            if obs.held_object is not None:
                assert len(csp.variables) == 8
                placement, surface, placement_base_pose = csp.variables[5:]
                placement_sampler = self._generate_placement_sampler(
                    obs.held_object, csp, placement, surface, placement_base_pose
                )
                samplers.append(placement_sampler)

            return samplers

        if self._current_mission == "put away robot held object":
            placement, surface, placement_base_pose = csp.variables
            placement_sampler = self._generate_placement_sampler(
                obs.held_object, obs, csp, placement, surface, placement_base_pose
            )
            return [placement_sampler]

        if self._current_mission == "put away human held object":
            grasp_base_pose, grasp_yaw, placement, surface, placement_base_pose = (
                csp.variables[:5]
            )

            def _sample_grasp_pose(
                _: dict[CSPVariable, Any], rng: np.random.Generator
            ) -> dict[CSPVariable, Any]:
                del rng  # not actually sampling right now, for simplicity
                base_pose = get_target_base_pose(obs, "bed", self._sim)
                yaw = np.array([-np.pi / 2])
                return {grasp_base_pose: base_pose, grasp_yaw: yaw}

            grasp_sampler = FunctionalCSPSampler(
                _sample_grasp_pose, csp, {grasp_base_pose, grasp_yaw}
            )

            assert obs.human_held_object is not None
            placement_sampler = self._generate_placement_sampler(
                obs.human_held_object, obs, csp, placement, surface, placement_base_pose
            )

            samplers = [grasp_sampler, placement_sampler]

            if obs.held_object is not None:
                assert len(csp.variables) == 8
                first_placement, first_surface, first_placement_base_pose = (
                    csp.variables[5:]
                )
                first_placement_sampler = self._generate_placement_sampler(
                    obs.held_object,
                    obs,
                    csp,
                    first_placement,
                    first_surface,
                    first_placement_base_pose,
                )
                samplers.append(first_placement_sampler)

            return samplers

        if self._current_mission == "clean":

            surfaces = sorted(self._sim.get_surface_names())
            surface, robot_state = csp.variables[:2]

            def _sample_surface(
                _: dict[CSPVariable, Any], rng: np.random.Generator
            ) -> dict[CSPVariable, Any]:
                surface_name = surfaces[rng.choice(len(surfaces))]
                surface_id = self._sim.get_object_id_from_name(surface_name)
                candidates = sorted(self._sim.get_surface_link_ids(surface_id))
                surface_link_id = candidates[rng.choice(len(candidates))]
                return {surface: (surface_name, surface_link_id)}

            surface_sampler = FunctionalCSPSampler(_sample_surface, csp, {surface})

            def _robot_state_sampler(
                sol: dict[CSPVariable, Any], rng: np.random.Generator
            ) -> dict[CSPVariable, Any] | None:
                # Sample base pose.
                dx, dy = rng.uniform([-0.1, -0.1], [0.1, 0.1])
                position = (
                    self._sim.scene_spec.robot_base_pose.position[0] + dx,
                    self._sim.scene_spec.robot_base_pose.position[1] + dy,
                    self._sim.scene_spec.robot_base_pose.position[2],
                )
                orientation = self._sim.scene_spec.robot_base_pose.orientation
                base_pose = Pose(position, orientation)
                # Sample joints.
                self._sim.set_robot_base(base_pose)
                surface_name, surface_link_id = sol[surface]
                num_rots = 1 if surface_name == "table" else 0
                ee_init_pose = self._get_prewipe_end_effector_pose(
                    surface_name, surface_link_id, num_rots
                )
                collision_ids = self._sim.get_collision_ids() - {
                    self._sim.current_held_object_id
                }
                # The number of calls to the RNG internally to the function is
                # nondeterministic, so make a new RNG to maintain determinism.
                ik_rng = create_rng_from_rng(rng)
                try:
                    joint_state = next(
                        sample_collision_free_inverse_kinematics(
                            self._sim.robot,
                            ee_init_pose,
                            collision_ids,
                            ik_rng,
                            held_object=self._sim.current_held_object_id,
                            base_link_to_held_obj=self._sim.current_grasp_transform,
                            max_time=1.0,  # way more than enough
                        )
                    )
                except StopIteration:
                    return None
                return {robot_state: (base_pose, np.array(joint_state))}

            robot_state_sampler = FunctionalCSPSampler(
                _robot_state_sampler, csp, {robot_state}
            )

            samplers = [surface_sampler, robot_state_sampler]

            if obs.held_object is not None:
                assert len(csp.variables) == 6
                placement, placement_surface, placement_base_pose = csp.variables[3:]
                placement_sampler = self._generate_placement_sampler(
                    obs.held_object,
                    obs,
                    csp,
                    placement,
                    placement_surface,
                    placement_base_pose,
                )
                samplers.append(placement_sampler)

            return samplers

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

        if self._current_mission == "put away robot held object":

            return _PutAwayRobotHeldObjectCSPPolicy(
                self._sim,
                csp,
                seed=self._seed,
                max_motion_planning_candidates=self._max_motion_planning_candidates,
            )

        if self._current_mission == "put away human held object":

            return _PutAwayHumanHeldObjectCSPPolicy(
                self._sim,
                csp,
                seed=self._seed,
                max_motion_planning_candidates=self._max_motion_planning_candidates,
            )

        if self._current_mission == "clean":

            return _CleanCSPPolicy(
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
            self._update_surface_can_be_cleaned(obs, next_obs)
        self._update_current_mission(obs)

    def _generate_surface_variable(
        self, surface_name: str = "surface"
    ) -> tuple[CSPVariable, Any]:
        # Choose a surface (and link ID on the surface).
        surfaces = sorted(self._sim.get_surface_names())
        surface = CSPVariable(
            surface_name, Tuple([EnumSpace(surfaces), Discrete(1000000, start=-1)])
        )

        init_surface = surfaces[self._rng.choice(len(surfaces))]
        init_surface_id = self._sim.get_object_id_from_name(init_surface)
        surface_link_ids = sorted(self._sim.get_surface_link_ids(init_surface_id))
        init_surface_link_id = surface_link_ids[self._rng.choice(len(surface_link_ids))]
        return surface, (init_surface, init_surface_link_id)

    def _generate_placement_variables(
        self,
        default_base_pose: Pose,
        placement_name: str = "placement",
        surface_name: str = "surface",
        placement_base_pose_name: str = "placement_base_pose",
    ) -> tuple[list[CSPVariable], dict[CSPVariable, Any]]:
        placement = CSPVariable(placement_name, PoseSpace())
        placement_init = Pose.identity()
        surface, surface_init = self._generate_surface_variable(surface_name)
        base_pose = CSPVariable(placement_base_pose_name, PoseSpace())
        variables = [placement, surface, base_pose]
        initialization = {
            placement: placement_init,
            surface: surface_init,
            base_pose: default_base_pose,
        }
        return variables, initialization

    def _generate_plan_to_place_exists_constraint(
        self,
        obs: PyBulletState,
        placement_var: CSPVariable,
        surface_var: CSPVariable,
        placement_base_pose_var: CSPVariable,
        constraint_name: str = "plan_to_place_exists",
    ) -> CSPConstraint:
        def _plan_to_place_exists(
            placement_pose: Pose,
            surface_name_and_link: tuple[str, int],
            base_pose: Pose,
        ) -> bool:
            assert obs.held_object is not None
            surface_name, surface_link_id = surface_name_and_link
            max_mp_candidates = self._max_motion_planning_candidates
            self._sim.set_state(obs)
            self._sim.set_robot_base(base_pose)
            obs_after_base_move = self._sim.get_state()
            plan = get_plan_to_place_object(
                obs_after_base_move,
                obs.held_object,
                surface_name,
                placement_pose,
                self._sim,
                max_motion_planning_time=1e-1,
                max_motion_planning_candidates=max_mp_candidates,
                surface_link_id=surface_link_id,
            )
            result = plan is not None
            return result

        plan_to_place_exists = FunctionalCSPConstraint(
            constraint_name,
            [placement_var, surface_var, placement_base_pose_var],
            _plan_to_place_exists,
        )

        return plan_to_place_exists

    def _generate_placement_sampler(
        self,
        held_object: str,
        obs: PyBulletState,
        csp: CSP,
        placement_var: CSPVariable,
        surface_var: CSPVariable,
        placement_base_pose_var: CSPVariable,
    ) -> CSPSampler:

        surfaces = sorted(self._sim.get_surface_names())

        held_object_id = self._sim.get_object_id_from_name(held_object)
        held_object_link_id = (
            self._sim.duster_head_link_id if held_object == "duster" else -1
        )
        held_obj_half_extents_at_placement = self._sim.get_default_half_extents(
            held_object_id, held_object_link_id
        )

        def _sample_placement(
            _: dict[CSPVariable, Any], rng: np.random.Generator
        ) -> dict[CSPVariable, Any]:
            surface_name = surfaces[rng.choice(len(surfaces))]
            surface_id = self._sim.get_object_id_from_name(surface_name)
            candidates = sorted(self._sim.get_surface_link_ids(surface_id))
            surface_link_id = candidates[rng.choice(len(candidates))]
            base_pose = get_target_base_pose(obs, surface_name, self._sim)
            placement_pose = next(
                generate_surface_placements(
                    surface_id,
                    held_obj_half_extents_at_placement,
                    rng,
                    self._sim.physics_client_id,
                    surface_link_id=surface_link_id,
                )
            )
            return {
                placement_var: placement_pose,
                surface_var: (surface_name, surface_link_id),
                placement_base_pose_var: base_pose,
            }

        placement_sampler = FunctionalCSPSampler(
            _sample_placement,
            csp,
            {placement_var, surface_var, placement_base_pose_var},
        )

        return placement_sampler

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
        # Don't learn from duster, which is not handed over.
        if obs.held_object == "duster":
            return
        # Only learn from cases where the robot triggered "hand over".
        if not np.isclose(act[0], 2):
            return
        assert act[1] == "Here you go!"
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
        # Only learn from attempted handovers, not cleaning.
        if next_obs.held_object == "duster":
            return
        # For now, only learn when the robot triggered hand over.
        if not np.isclose(act[0], 2):
            return
        assert act[1] == "Here you go!"
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

    def _update_surface_can_be_cleaned(
        self, obs: PyBulletState, next_obs: PyBulletState
    ) -> None:
        # Only update when holding the duster.
        if obs.held_object != "duster":
            return
        # Wait while retraction happens.
        if (
            self._steps_since_last_cleaning_admonishment is not None
            and self._steps_since_last_cleaning_admonishment
            <= self._sim.scene_spec.cleaning_admonishment_min_time_interval
        ):
            return
        # Figure out which surface, if any, was touched based on patch change.
        surface_wiped: tuple[str, int] | None = None
        for surf in obs.surface_dust_patches:
            old_patch_arr = obs.surface_dust_patches[surf]
            new_patch_arr = next_obs.surface_dust_patches[surf]
            old_clean = np.isclose(old_patch_arr, 0.0)
            new_clean = np.isclose(new_patch_arr, 0.0)
            if np.any(new_clean & ~old_clean):
                assert surface_wiped is None
                surface_wiped = surf
        human_admonished = (
            next_obs.human_text is not None and "Don't clean" in next_obs.human_text
        )
        if human_admonished:
            assert surface_wiped is not None
            self._steps_since_last_cleaning_admonishment = 0
        else:
            if self._steps_since_last_cleaning_admonishment is not None:
                self._steps_since_last_cleaning_admonishment += 1
        if surface_wiped is not None:
            # If the human complained, we should not clean anymore.
            can_clean = not human_admonished
            if self._surface_can_be_cleaned[surface_wiped] == "unknown":
                self._surface_can_be_cleaned[surface_wiped] = can_clean
                logging.info(
                    f"Updated can-clean status for {surface_wiped}: " f"{can_clean}"
                )
            assert self._surface_can_be_cleaned[surface_wiped] == can_clean

    def _update_current_mission(self, obs: PyBulletState) -> None:
        mission = _infer_mission_from_obs(obs)
        if mission is not None:
            self._current_mission = mission

    def _snap_duster_to_end_effector(self) -> Pose:
        grasp_pose = self._sim.scene_spec.duster_grasp
        end_effector_pose = self._sim.robot.get_end_effector_pose()
        duster_pose = multiply_poses(end_effector_pose, grasp_pose.invert())
        set_pose(self._sim.duster_id, duster_pose, self._sim.physics_client_id)
        world_to_duster_head = get_link_pose(
            self._sim.duster_id,
            self._sim.duster_head_link_id,
            self._sim.physics_client_id,
        )
        ee_to_duster_head = multiply_poses(
            end_effector_pose.invert(), world_to_duster_head
        )
        self._sim.current_held_object_id = self._sim.duster_id
        self._sim.current_grasp_transform = grasp_pose.invert()
        return ee_to_duster_head

    def _get_prewipe_end_effector_pose(
        self, surface_name: str, surface_link_id: int, num_rots: int
    ) -> Pose:
        ee_to_duster_head = self._snap_duster_to_end_effector()
        grasping_state = self._sim.get_state()
        duster_head_plan = get_duster_head_frame_wiping_plan(
            grasping_state,
            "duster",
            surface_name,
            num_rots,
            self._sim,
            surface_link_id=surface_link_id,
        )
        return multiply_poses(duster_head_plan[0], ee_to_duster_head.invert())

    def get_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if isinstance(self._rom_model, TrainableROMModel):
            metrics.update(self._rom_model.get_metrics())
        for book_description in self._sim.book_descriptions:
            lp = self._book_is_preferred_logprob(book_description)
            entropy = bernoulli_entropy(lp)
            metrics[f"entropy-{book_description}"] = entropy
        return metrics
