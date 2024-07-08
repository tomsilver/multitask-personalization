"""Experiment varying intake horizon."""

import os
from pathlib import Path

import imageio.v2 as iio
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from multitask_personalization.envs.grid_world import (
    _EMPTY,
    _OBSTACLE,
    GridTask,
)
from multitask_personalization.methods.approach import Approach
from multitask_personalization.methods.calibration.calibrator import Calibrator
from multitask_personalization.methods.calibration.grid_world_calibrator import (
    GridWorldCalibrator,
    OracleGridWorldCalibrator,
)
from multitask_personalization.methods.interaction.random_interaction import (
    RandomInteractionMethod,
)
from multitask_personalization.methods.policies.grid_world_policy import (
    GridWorldParameterizedPolicy,
)
from multitask_personalization.structs import Image


def _main(
    start_seed: int,
    num_seeds: int,
    num_tasks: int,
    num_coins: int,
    outdir: Path,
    load: bool,
    make_videos: bool,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    csv_file = outdir / "grid_experiment.csv"
    if load:
        assert csv_file.exists()
        df = pd.read_csv(csv_file)
        return _df_to_plot(df, outdir)
    columns = ["Seed", "Approach", "Num Intake Steps", "Returns"]
    approaches = ["Oracle", "Random"]
    all_num_intake_steps = [0, 10, 100, 500, 1000]
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
                    num_tasks,
                    num_coins,
                    num_intake_steps,
                    make_videos,
                    outdir,
                )
                results.append((seed, approach, num_intake_steps, returns))
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(csv_file)
    return _df_to_plot(df, outdir)


def _df_to_plot(df: pd.DataFrame, outdir: Path) -> None:
    matplotlib.rcParams.update({"font.size": 20})
    fig_file = outdir / "grid_experiment.png"

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
    plt.title("Grid World")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_file, dpi=150)
    print(f"Wrote out to {fig_file}")


def _sample_terminal_rewards(
    terminal_locs: list[tuple[int, int]], rng: np.random.Generator
) -> dict[tuple[int, int], float]:
    rew_list = list(rng.normal(size=len(terminal_locs)))
    return dict(zip(terminal_locs, rew_list))


def _sample_coin_weights(num_coins: int, rng: np.random.Generator) -> list[float]:
    return list(rng.uniform(size=num_coins))


def _run_single(
    seed: int,
    approach_name: str,
    num_tasks: int,
    num_coins: int,
    num_intake_steps: int,
    make_videos: bool,
    outdir: Path,
) -> float:

    # Create things that are constant in the world.
    E, O, X, Y = _EMPTY, _OBSTACLE, 2, 3
    grid = np.array(
        [
            [E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, X],
            [E, E, O, O, O, Y, E, E, E, E, E, O, E, E, X, O, O, O, E, E],
            [X, O, O, E, O, E, E, O, O, O, E, E, E, E, E, O, E, O, E, E],
            [E, E, E, E, O, O, E, E, E, O, X, O, E, E, O, O, E, E, E, Y],
            [E, E, E, E, E, E, Y, E, E, O, E, O, E, E, E, E, E, E, E, E],
            [E, O, E, E, E, E, E, E, E, E, E, O, O, E, Y, E, X, E, E, E],
            [E, O, E, E, O, O, O, E, E, X, O, E, E, E, E, E, E, E, E, X],
            [E, E, E, X, E, E, E, E, E, E, E, E, E, E, E, E, Y, E, E, E],
        ]
    )
    agnostic_locs = np.argwhere(grid == X)
    specific_locs = np.argwhere(grid == Y)
    grid[np.logical_or(grid == X, grid == Y)] = E
    terminal_types = {tuple(l): "agnostic" for l in agnostic_locs}
    terminal_types.update({tuple(l): "specific" for l in specific_locs})
    terminal_locs = sorted(terminal_types)
    initial_state = (0, 0)

    # Create things that are task-specific.
    tasks: list[GridTask] = []
    rng = np.random.default_rng(seed)
    for i in range(num_tasks):
        task_id = f"task{i}"
        terminal_rewards = _sample_terminal_rewards(terminal_locs, rng)
        coin_weights = _sample_coin_weights(num_coins, rng)
        task = GridTask(
            task_id,
            grid,
            terminal_rewards,
            initial_state,
            terminal_types,
            coin_weights,
            num_intake_steps,
        )
        tasks.append(task)

    # Create the approach.
    if approach_name == "Random":
        calibrator: Calibrator = GridWorldCalibrator(terminal_locs)
        im = RandomInteractionMethod(seed=seed)
    elif approach_name == "Oracle":
        calibrator = OracleGridWorldCalibrator(tasks)
        im = RandomInteractionMethod(seed=seed)
    else:
        raise NotImplementedError
    policy = GridWorldParameterizedPolicy(grid, terminal_locs)
    approach = Approach(calibrator, im, policy)

    # Go through each task.
    returns = 0.0
    imgs: list[Image] = []
    for task in tasks:
        # Run the intake process.
        ip = task.intake_process
        rng = np.random.default_rng(seed)
        approach.reset(task.id, ip.action_space, ip.observation_space)
        for _ in range(ip.horizon):
            act = approach.get_intake_action()
            obs = ip.sample_next_observation(act, rng)
            approach.record_intake_observation(obs)
        approach.finish_intake()

        # Run the MDP.
        mdp = task.mdp
        rng = np.random.default_rng(seed)
        state = mdp.sample_initial_state(rng)
        if make_videos:
            imgs.append((mdp.render_state(state)))
        for _ in range(10000):  # should be more than enough
            if mdp.state_is_terminal(state):
                break
            action = approach.get_mdp_action(state)
            next_state = mdp.sample_next_state(state, action, rng)
            returns += mdp.get_reward(state, action, next_state)
            state = next_state
            if make_videos:
                imgs.append((mdp.render_state(state)))

    if make_videos:
        video_file = outdir / f"grid_experiment_{seed}_{approach_name}.gif"
        iio.mimsave(video_file, imgs)  # type: ignore
        print(f"Wrote out to {video_file}")

    return returns


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_seeds", default=10, type=int)
    parser.add_argument("--num_tasks", default=10, type=int)
    parser.add_argument("--num_coins", default=100, type=int)
    parser.add_argument("--outdir", default=Path("results"), type=Path)
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--make_videos", action="store_true")
    parser_args = parser.parse_args()
    _main(
        parser_args.seed,
        parser_args.num_seeds,
        parser_args.num_tasks,
        parser_args.num_coins,
        parser_args.outdir,
        parser_args.load,
        parser_args.make_videos,
    )
