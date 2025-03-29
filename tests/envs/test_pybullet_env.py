"""Tests for pybullet_env.py."""

import os
from pathlib import Path

import gymnasium as gym

import numpy as np
import pytest
from tomsutils.llm import OpenAILLM
from pybullet_helpers.geometry import Pose
from multitask_personalization.envs.pybullet.pybullet_human import SmoothHumanSpec

from multitask_personalization.envs.pybullet.pybullet_env import PyBulletEnv
from multitask_personalization.envs.pybullet.pybullet_scene_spec import (
    HiddenSceneSpec,
    PyBulletSceneSpec,
)
from multitask_personalization.envs.pybullet.pybullet_structs import PyBulletState
from multitask_personalization.envs.pybullet.pybullet_utils import PyBulletCannedLLM
from multitask_personalization.rom.models import SphericalROMModel

_LLM_CACHE_DIR = Path(__file__).parents[1] / "unit_test_llm_cache"
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "NOT A REAL KEY"  # will not be used


@pytest.mark.parametrize(
    "llm",
    [
        OpenAILLM(
            model_name="gpt-4o-mini",
            cache_dir=_LLM_CACHE_DIR,
            max_tokens=700,
            use_cache_only=True,
        ),
        PyBulletCannedLLM(_LLM_CACHE_DIR),
    ],
)
def test_pybullet_env(llm):
    """Tests for pybullet_env.py."""
    seed = 123

    scene_spec = PyBulletSceneSpec(num_books=3)
    book_preferences = "I like pretty much anything!"
    rom_model = SphericalROMModel(scene_spec.human_spec)
    surfaces_robot_can_clean = [
        ("table", -1),
        ("shelf", 0),
        ("shelf", 1),
        ("shelf", 2),
    ]
    hidden_spec = HiddenSceneSpec(
        missions="all",
        book_preferences=book_preferences,
        rom_model=rom_model,
        surfaces_robot_can_clean=surfaces_robot_can_clean,
    )
    env = PyBulletEnv(
        scene_spec,
        llm,
        hidden_spec=hidden_spec,
        use_gui=False,
        seed=seed,
    )
    env.action_space.seed(seed)
    obs, _ = env.reset()
    assert isinstance(obs, PyBulletState)

    for _ in range(10):
        act = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(act)
        assert np.isclose(reward, 0.0)
        assert isinstance(obs, PyBulletState)
        if terminated:
            break
        assert not truncated

    env.close()


# TODO extract this if it works
class RecordVideoWithIntermediateFrames(gym.wrappers.RecordVideo):

    def _capture_frame(self):
        if hasattr(self.env, "interstates") and hasattr(self.env, "set_state"):
            # Capture intermediate frames from the environment's interstates.
            state = self.env.get_state()
            for interstate in self.env.interstates:
                self.env.set_state(interstate)
                # Call the original method to capture the frame.
                super()._capture_frame()
            self.env.set_state(state)
        return super()._capture_frame()



def test_pybullet_human_handover():
    """Tests for handing the human a book."""
    seed = 123
    llm = PyBulletCannedLLM(_LLM_CACHE_DIR)
    scene_spec = PyBulletSceneSpec(num_books=3, num_side_tables=1,
                                   human_spec=SmoothHumanSpec(),
                                   use_default_camera_kwargs=True)
    book_preferences = "I like pretty much anything!"
    rom_model = SphericalROMModel(scene_spec.human_spec)
    surfaces_robot_can_clean = [
        ("table", -1),
        ("shelf", 0),
        ("shelf", 1),
        ("shelf", 2),
    ]
    hidden_spec = HiddenSceneSpec(
        missions="all",
        book_preferences=book_preferences,
        rom_model=rom_model,
        surfaces_robot_can_clean=surfaces_robot_can_clean,
    )
    env = PyBulletEnv(
        scene_spec,
        llm,
        hidden_spec=hidden_spec,
        use_gui=True,
        seed=seed,
    )

    # TODO
    env = RecordVideoWithIntermediateFrames(
        env, "test_handover"
    )

    env.action_space.seed(seed)
    obs, _ = env.reset()
    assert isinstance(obs, PyBulletState)

    # Create a state where the robot is prepared to hand over a book.
    pre_handover_state = PyBulletState(robot_base=Pose(position=(0.999, 0.1998, 0.0), orientation=(0.0, -0.0, -7.709745172984898e-18, 1.0)), robot_joints=[1.6480722194817055, 1.3195409286293214, -2.220582845782344, 2.1401371896226644, 0.7169854126804442, 0.37902830854230807, 0.23839397796035167, 0.3, 0.3, 0.3, 0.3, -0.3, -0.3], human_base=Pose(position=(1.7599999904632568, 0.5779999494552612, 0.6499999761581421), orientation=(0.0, 0.0, 0.0, 1.0)), human_joints=[0.0, 0.1, 0.1, -1.08786023, 0.0, 0.0], cup_pose=Pose(position=(-1000.0, -1000.0, 0.05), orientation=(0.0, 0.0, 0.0, 1.0)), duster_pose=Pose(position=(-0.75, 0.2, 0.04), orientation=(0.0, 0.0, 0.0, 1.0)), book_poses=[Pose(position=(-0.2, 0.75, 0.12999999999999998), orientation=(0.0, 0.0, 0.0, 1.0)), Pose(position=(1.3937150239944458, 0.18239983916282654, 0.4925205707550049), orientation=(0.27058991981691144, 0.6532875941824325, -0.27057278348141683, 0.653289203507853)), Pose(position=(0.2, 0.75, 0.12999999999999998), orientation=(0.0, 0.0, 0.0, 1.0))], book_descriptions=['Title: Book 0. Author: Love.', 'Title: Book 1. Author: Love.', 'Title: Book 2. Author: Hate.'], grasp_transform=Pose(position=(1.5525260096183047e-05, 4.564225673675537e-05, 0.0014103055000305176), orientation=(0.7070915699005127, -3.4800884805008536e-06, 4.549259756458923e-06, 0.7071219086647034)), surface_dust_patches={('shelf', 0): np.array([[1., 1.],
       [1., 1.]]), ('shelf', 1): np.array([[1., 1.],
       [1., 1.]]), ('shelf', 2): np.array([[1., 1.],
       [1., 1.]]), ('side-table-0', -1): np.array([[1., 1.],
       [1., 1.]]), ('table', -1): np.array([[1., 1.],
       [1., 1.]])}, held_object='Title: Book 1. Author: Love.', human_text=None, human_held_object=None, human_grasp_transform=None)
    env.unwrapped.set_state(pre_handover_state)
    env.step((2, "Here you go!"))
    env.close()

    # from pybullet_helpers.inverse_kinematics import inverse_kinematics
    # from pybullet_helpers.geometry import rotate_pose
    # from pybullet_helpers.gui import visualize_pose
    # handover_pose = rotate_pose(env.robot.get_end_effector_pose(), roll=np.pi)
    # visualize_pose(handover_pose, env.physics_client_id)
    # visualize_pose(env.human.get_end_effector_pose(), env.physics_client_id)
    # joints = inverse_kinematics(env.human, handover_pose)

    # import pybullet as p
    # while True:
    #     p.getMouseEvents(env.physics_client_id)