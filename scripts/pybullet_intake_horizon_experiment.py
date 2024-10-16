"""Experiment varying intake horizon."""

import os
from pathlib import Path

import imageio.v2 as iio
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from multitask_personalization.envs.pybullet.pybullet_tasks import (
    PyBulletTask,
)
from multitask_personalization.methods.approach import Approach
from multitask_personalization.methods.calibration.calibrator import Calibrator
from multitask_personalization.methods.calibration.pybullet_calibrator import (
    OraclePyBulletCalibrator,
    PyBulletCalibrator,
)
from multitask_personalization.methods.interaction.random_interaction import (
    RandomInteractionMethod,
)
from multitask_personalization.methods.policies.pybullet_policy import (
    PyBulletParameterizedPolicy,
)
from multitask_personalization.structs import Image


def _main(
    start_seed: int,
    num_seeds: int,
    outdir: Path,
    load: bool,
    make_videos: bool,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    csv_file = outdir / "pybullet_experiment.csv"
    if load:
        assert csv_file.exists()
        df = pd.read_csv(csv_file)
        return _df_to_plot(df, outdir)
    columns = ["Seed", "Approach", "Num Intake Steps", "Returns"]
    approaches = [
        "Random",
        "Oracle",
    ]
    all_num_intake_steps = [0, 10, 50, 100, 250]
    results: list[tuple[int, str, int, float]] = []
    for seed in range(start_seed, start_seed + num_seeds):
        print(f"Starting {seed=}")
        for approach in approaches:
            print(f"Starting {approach=}")
            for num_intake_steps in all_num_intake_steps:
                print(f"Starting {num_intake_steps=}")
                returns = _run_single(
                    seed,
                    approach,
                    num_intake_steps,
                    make_videos,
                    outdir,
                )
                print(f"Returns: {returns}")
                results.append((seed, approach, num_intake_steps, returns))
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(csv_file)
    return _df_to_plot(df, outdir)


def _df_to_plot(df: pd.DataFrame, outdir: Path) -> None:
    matplotlib.rcParams.update({"font.size": 20})
    fig_file = outdir / "pybullet_experiment.png"

    grouped = df.groupby(["Num Intake Steps", "Approach"]).agg(
        {"Returns": ["mean", "sem"]}
    )
    grouped.columns = grouped.columns.droplevel(0)
    grouped = grouped.rename(columns={"mean": "Returns_mean", "sem": "Returns_sem"})
    grouped = grouped.reset_index()
    plt.figure(figsize=(10, 6))

    for approach in grouped["Approach"].unique():
        approach_data = grouped[grouped["Approach"] == approach]
        plt.plot(
            approach_data["Num Intake Steps"],
            approach_data["Returns_mean"],
            label=approach,
        )
        plt.fill_between(
            approach_data["Num Intake Steps"],
            approach_data["Returns_mean"] - approach_data["Returns_sem"],
            approach_data["Returns_mean"] + approach_data["Returns_sem"],
            alpha=0.2,
        )

    plt.xlabel("# Intake Steps")
    plt.ylabel("Evaluation Performance")
    plt.title("Simulated Handover")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_file, dpi=150)
    print(f"Wrote out to {fig_file}")


def _run_single(
    seed: int,
    approach_name: str,
    num_intake_steps: int,
    make_videos: bool,
    outdir: Path,
) -> float:

    task = PyBulletTask(intake_horizon=num_intake_steps, use_gui=False)

    # Create the approach.
    if approach_name == "Random":
        calibrator: Calibrator = PyBulletCalibrator(task.scene_description)
        im = RandomInteractionMethod(seed=seed)
    elif approach_name == "Oracle":
        calibrator = OraclePyBulletCalibrator(task.scene_description)
        im = RandomInteractionMethod(seed=seed)
    else:
        raise NotImplementedError
    policy = PyBulletParameterizedPolicy(task.scene_description, seed=seed)
    approach = Approach(calibrator, im, policy)

    returns = 0.0
    imgs: list[Image] = []

    # Run the intake process.
    ip = task.intake_process
    rng = np.random.default_rng(seed)
    ip.action_space.seed(seed)
    approach.reset(task.id, ip.action_space, ip.observation_space)
    for _ in range(ip.horizon):
        act = approach.get_intake_action()
        obs = ip.sample_next_observation(act, rng)
        approach.record_intake_observation(obs)
    approach.finish_intake()

    # Run the MDP.
    mdp = task.mdp
    rng = np.random.default_rng(seed)
    mdp.action_space.seed(seed)
    state = mdp.sample_initial_state(rng)
    if make_videos:
        imgs.append((mdp.render_state(state)))
    for _ in range(250):  # should be more than enough
        if mdp.state_is_terminal(state):
            break
        action = approach.get_mdp_action(state)
        next_state = mdp.sample_next_state(state, action, rng)
        returns += mdp.get_reward(state, action, next_state)
        state = next_state
        if make_videos:
            imgs.append((mdp.render_state(state)))

    if make_videos:
        video_file = outdir / f"pybullet_experiment_{seed}_{approach_name}.gif"
        iio.mimsave(video_file, imgs)  # type: ignore
        print(f"Wrote out to {video_file}")

    task.close()

    return returns


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_seeds", default=5, type=int)
    parser.add_argument("--outdir", default=Path("results"), type=Path)
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--make_videos", action="store_true")
    parser_args = parser.parse_args()
    _main(
        parser_args.seed,
        parser_args.num_seeds,
        parser_args.outdir,
        parser_args.load,
        parser_args.make_videos,
    )
