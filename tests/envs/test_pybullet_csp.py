"""Tests for pybullet_csp.py."""

import os
from pathlib import Path

import numpy as np
from tomsutils.llm import OpenAILLM

from multitask_personalization.csp_solvers import RandomWalkCSPSolver
from multitask_personalization.envs.pybullet.pybullet_csp import (
    PyBulletCSPGenerator,
)
from multitask_personalization.envs.pybullet.pybullet_env import PyBulletEnv
from multitask_personalization.envs.pybullet.pybullet_scene_spec import (
    HiddenSceneSpec,
    PyBulletSceneSpec,
)
from multitask_personalization.envs.pybullet.pybullet_structs import (
    PyBulletState,
)
from multitask_personalization.rom.models import SphericalROMModel


def test_pybullet_csp():
    """Tests for pybullet_csp.py."""
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "NOT A REAL KEY"  # will not be used
    seed = 123
    default_scene_spec = PyBulletSceneSpec()
    scene_spec = PyBulletSceneSpec(
        book_half_extents=default_scene_spec.book_half_extents[:3],
        book_poses=default_scene_spec.book_poses[:3],
        book_rgbas=default_scene_spec.book_rgbas[:3],
    )
    book_preferences = "I like pretty much anything!"
    rom_model = SphericalROMModel(
        scene_spec.human_spec, min_possible_radius=0.49, max_possible_radius=0.51
    )
    hidden_spec = HiddenSceneSpec(
        book_preferences=book_preferences, rom_model=rom_model
    )

    llm = OpenAILLM(
        model_name="gpt-4o-mini",
        cache_dir=Path(__file__).parents[1] / "unit_test_llm_cache",
        max_tokens=700,
        use_cache_only=True,
    )

    # Create a real environment.
    env = PyBulletEnv(
        scene_spec,
        llm,
        hidden_spec=hidden_spec,
        use_gui=False,
        seed=seed,
    )

    # Uncomment to create video.
    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/test-pybullet-csp")

    env.action_space.seed(seed)
    obs, _ = env.reset()
    assert isinstance(obs, PyBulletState)

    # Create a simulator.
    sim = PyBulletEnv(scene_spec, llm, use_gui=False, seed=seed)

    # Create the CSP.
    csp_generator = PyBulletCSPGenerator(
        sim,
        rom_model,
        llm,
        seed=seed,
        book_preference_initialization="I like everything!",
    )

    solver = RandomWalkCSPSolver(
        seed, min_num_satisfying_solutions=1, show_progress_bar=False
    )

    # Generate and solve CSPs twice in a row. The first time will be a book
    # handover and the second time will be a placement.
    for _ in range(2):
        csp, samplers, policy, initialization = csp_generator.generate(obs)

        # Solve the CSP.
        sol = solver.solve(
            csp,
            initialization,
            samplers,
        )
        assert sol is not None
        policy.reset(sol)

        # Run the policy.
        for _ in range(1000):
            act = policy.step(obs)
            obs, reward, terminated, truncated, _ = env.step(act)
            assert isinstance(obs, PyBulletState)
            assert np.isclose(reward, 0.0)
            if policy.check_termination(obs):
                break
            assert not terminated
            assert not truncated
        else:
            assert False, "Policy did not terminate."

    env.close()
