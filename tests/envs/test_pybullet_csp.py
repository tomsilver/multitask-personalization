"""Tests for pybullet_csp.py."""

from pathlib import Path

import numpy as np

from multitask_personalization.csp_solvers import RandomWalkCSPSolver
from multitask_personalization.envs.pybullet.pybullet_csp import (
    PyBulletCSPGenerator,
)
from multitask_personalization.envs.pybullet.pybullet_utils import PyBulletCannedLLM
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
    seed = 123
    default_scene_spec = PyBulletSceneSpec()
    scene_spec = PyBulletSceneSpec(
        book_half_extents=default_scene_spec.book_half_extents[:3],
        book_poses=default_scene_spec.book_poses[:3],
    )
    book_preferences = "I like pretty much anything!"
    rom_model = SphericalROMModel(
        scene_spec.human_spec, min_possible_radius=0.49, max_possible_radius=0.51
    )
    surfaces_robot_can_clean = [
        ("table", -1),
        ("shelf", 0),
        ("shelf", 1),
        ("shelf", 2),
    ]
    hidden_spec = HiddenSceneSpec(
        book_preferences=book_preferences,
        rom_model=rom_model,
        surfaces_robot_can_clean=surfaces_robot_can_clean,
    )

    llm = PyBulletCannedLLM(
        cache_dir=Path(__file__).parents[1] / "unit_test_llm_cache",
    )

    # Create a real environment.
    env = PyBulletEnv(
        scene_spec,
        llm,
        hidden_spec=hidden_spec,
        use_gui=True,
        seed=seed,
    )
    env.action_space.seed(seed)

    # Uncomment to create video.
    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/test-pybullet-csp")

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

    all_missions = env._create_possible_missions()  # pylint: disable=protected-access
    mission_id_to_mission = {m.get_id(): m for m in all_missions}
    book_handover_mission = mission_id_to_mission["book handover"]
    clean_mission = mission_id_to_mission["clean"]

    def _run_mission(mission):
        # Override the mission and regenerate the observation.
        env.current_human_text = None
        env._reset_mission(mission)  # pylint: disable=protected-access
        obs = env.get_state()

        # Generate CSP.
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
        for _ in range(1000):  # should be more than enough
            act = policy.step(obs)
            if mission.check_complete(obs, act):
                break
            obs, reward, _, _, _ = env.step(act)
            assert isinstance(obs, PyBulletState)
            assert np.isclose(reward, 0.0)
            assert not policy.check_termination(obs)
        else:
            assert False, "Mission did not complete."

    # Save states to unit-test directory.
    saved_state_dir = Path(__file__).parents[1] / "unit_test_saved_states"
    assert saved_state_dir.exists()

    # Reset environment once.
    env.reset()

    # Uncomment to test from custom saved state.
    # custom_saved_state_fp = Path("...")
    # env.load_state(custom_saved_state_fp)
    # _run_mission(clean_mission)

    # Start with book handover.
    post_book_handover1_state_fp = saved_state_dir / "book_handover_1.p"
    _run_mission(book_handover_mission)
    env.save_state(post_book_handover1_state_fp)

    # Clean.
    env.load_state(post_book_handover1_state_fp)
    post_clean1_state_fp = saved_state_dir / "clean_1.p"
    _run_mission(clean_mission)
    env.save_state(post_clean1_state_fp)

    # Uncomment for more thorough tests (but too slow to merge).

    # # Clean again.
    # env.load_state(post_clean1_state_fp)
    # post_clean2_state_fp = saved_state_dir / "clean_2.p"
    # _run_mission(clean_mission)
    # env.save_state(post_clean2_state_fp)

    # # Get another book.
    # env.load_state(post_clean2_state_fp)
    # post_book_handover2_state_fp = saved_state_dir / "book_handover_2.p"
    # _run_mission(book_handover_mission)
    # env.save_state(post_book_handover2_state_fp)

    # # Get another book.
    # env.load_state(post_book_handover2_state_fp)
    # post_book_handover3_state_fp = saved_state_dir / "book_handover_3.p"
    # _run_mission(book_handover_mission)
    # env.save_state(post_book_handover3_state_fp)

    # # Get another book.
    # env.load_state(post_book_handover3_state_fp)
    # post_book_handover4_state_fp = saved_state_dir / "book_handover_4.p"
    # _run_mission(book_handover_mission)
    # env.save_state(post_book_handover4_state_fp)

    # # Clean.
    # env.load_state(post_book_handover4_state_fp)
    # post_clean3_state_fp = saved_state_dir / "clean_3.p"
    # _run_mission(clean_mission)
    # env.save_state(post_clean3_state_fp)

    # # Get another book.
    # env.load_state(post_clean3_state_fp)
    # post_book_handover_5_state_fp = saved_state_dir / "book_handover_5.p"
    # _run_mission(book_handover_mission)
    # env.save_state(post_book_handover_5_state_fp)

    env.close()
