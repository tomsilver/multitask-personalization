"""Specific missions for the robot in the PyBullet environment."""

import numpy as np
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from tomsutils.llm import LargeLanguageModel

from multitask_personalization.envs.pybullet.pybullet_structs import (
    PyBulletAction,
    PyBulletMission,
    PyBulletState,
)
from multitask_personalization.envs.pybullet.pybullet_utils import user_would_enjoy_book
from multitask_personalization.rom.models import ROMModel


class HandOverBookMission(PyBulletMission):
    """Hand over a book that the user enjoys."""

    def __init__(
        self,
        book_descriptions: list[str],
        sim_robot: FingeredSingleArmPyBulletRobot,
        rom_model: ROMModel,
        hidden_book_preferences: str,
        llm: LargeLanguageModel,
        seed: int,
    ) -> None:
        super().__init__()
        self._book_descriptions = book_descriptions
        self._robot = sim_robot
        self._rom_model = rom_model
        self._hidden_book_preferences = hidden_book_preferences
        self._llm = llm
        self._seed = seed

    def get_id(self) -> str:
        return "book handover"

    def get_mission_command(self) -> str:
        # Could add some variation with an LLM later.
        return "Please bring me a book to read"

    def check_initiable(self, state: PyBulletState) -> bool:
        return state.held_object is None

    def check_complete(self, state: PyBulletState, action: PyBulletAction) -> bool:
        robot_indicated_done = bool(np.isclose(action[0], 2))
        reachable = self._check_reachable(state)
        return reachable and robot_indicated_done

    def step(
        self, state: PyBulletState, action: PyBulletAction
    ) -> tuple[str | None, float]:
        robot_indicated_done = np.isclose(action[0], 2)
        # Robot needs to indicate done for the handover task.
        if not robot_indicated_done:
            return None, 0.0
        # Must be holding a book.
        if state.held_object not in self._book_descriptions:
            return None, -1.0
        book_description = state.held_object
        # Check if the book is reachable.
        if not self._check_reachable(state):
            return "I can't reach there", -1.0
        # Should be holding a preferred book.
        if not user_would_enjoy_book(
            book_description,
            self._hidden_book_preferences,
            self._llm,
            seed=self._seed,
        ):
            # The robot is attempting to hand over a book, but the user
            # doesn't actually like that book. Have the user explain in
            # natural language why they don't like the book.
            text = _explain_user_book_preference(
                book_description,
                self._hidden_book_preferences,
                self._llm,
                enjoyed=False,
                seed=self._seed,
            )
            return text, -1.0
        # The robot is successful in handing over the book. Have the user
        # elaborate on why they like this book.
        text = _explain_user_book_preference(
            book_description,
            self._hidden_book_preferences,
            self._llm,
            enjoyed=True,
            seed=self._seed,
        )
        return text, 1.0

    def _check_reachable(self, state: PyBulletState) -> bool:
        end_effector_pose = self._robot.forward_kinematics(state.robot_joints)
        return self._rom_model.check_position_reachable(
            np.array(end_effector_pose.position)
        )


class StoreHeldObjectMission(PyBulletMission):
    """Put away the thing the robot is holding."""

    def get_id(self) -> str:
        return "store held object"

    def get_mission_command(self) -> str:
        # Could add some variation with an LLM later.
        return "Put away the thing you're holding"

    def check_initiable(self, state: PyBulletState) -> bool:
        return state.held_object is not None

    def check_complete(self, state: PyBulletState, action: PyBulletAction) -> bool:
        return state.held_object is None

    def step(
        self, state: PyBulletState, action: PyBulletAction
    ) -> tuple[str | None, float]:
        return None, 0.0


class CleanSurfacesMission(PyBulletMission):
    """Clean some dirty surfaces."""

    def get_id(self) -> str:
        return "clean"

    def get_mission_command(self) -> str:
        # Could add some variation with an LLM later.
        return "Clean the dirty surfaces"

    def check_initiable(self, state: PyBulletState) -> bool:
        return state.held_object is None

    def check_complete(self, state: PyBulletState, action: PyBulletAction) -> bool:
        robot_indicated_done = bool(np.isclose(action[0], 2))
        return robot_indicated_done

    def step(
        self, state: PyBulletState, action: PyBulletAction
    ) -> tuple[str | None, float]:
        # Coming soon: penalty and feedback if the robot tries to touch an object
        # that it should not touch.
        return None, 0.0


def _explain_user_book_preference(
    book_description: str,
    user_preferences: str,
    llm: LargeLanguageModel,
    llm_temperature: float = 0.0,
    enjoyed: bool = False,
    seed: int = 0,
) -> str:
    """Have the user explain why they do or do not like the book."""
    # pylint: disable=line-too-long
    do_or_do_not_enjoy = "DO" if enjoyed else "DO NOT"
    prompt = f"""Pretend you are a human user with the following preferences about books:
    
User preferences: {user_preferences}

A robot is handing you the following book:

Book description: {book_description}

You want to explain to the robot why you {do_or_do_not_enjoy} enjoy this book.

Do not directly reveal the user preferences.

Return short dialogue as if you were the human user. Return only this. Do not explain anything."""

    response = llm.sample_completions(
        prompt,
        imgs=None,
        temperature=llm_temperature,
        seed=seed,
    )[0]
    return response
