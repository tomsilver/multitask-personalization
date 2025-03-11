"""Plot prediction error over time for learned ROM models."""

import argparse
from pathlib import Path
import pandas as pd
import os

import seaborn as sns
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

from plot_main_results import ENV_TO_DISPLAY_NAME, APPROACH_TO_DISPLAY_NAME, APPROACH_TO_COLOR


def _main( results_dir: Path, outfile: Path, config_filename: str = "config.yaml") -> None:
    """Creates and saves a plot."""
    env_name = "pybullet"

    # Gather the data.
    columns = ("Approach", "Seed", "Training Execution Time", "Error")
    results = []
    for subdir in os.listdir(results_dir):
        subdir_path = results_dir / subdir
        if os.path.isdir(subdir_path) and subdir_path.name.isdigit():
            model_dir = subdir_path / "models"
            config_path = subdir_path / config_filename
            cfg = OmegaConf.load(config_path)
            assert isinstance(cfg, DictConfig)
            assert cfg["env_name"] == env_name
            approach = cfg["approach_name"]
            seed = cfg["seed"]
            for checkpoint_dir in sorted(
                [d for d in model_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                key=lambda d: int(d.name),
            ):
                num_training_steps = int(checkpoint_dir.name)
                training_time = num_training_steps * cfg.env.dt
                error = 0.0  # TODO
                result = (approach, seed, training_time, error)
                results.append(result)
    df = pd.DataFrame(results, columns=columns)

    # Create the plot.
    plt.style.use(Path(__file__).parent / "custom.mplstyle")
    _, ax = plt.subplots()
    env_display_name = ENV_TO_DISPLAY_NAME[env_name]
    ax.set_title(f"{env_display_name}: ROM Prediction Error")
    ax.set_xlabel("Simulated Execution Time")
    ax.set_ylabel("ROM Prediction Error")
    for approach in sorted(set(df["Approach"])):
        approach_df = df[df["Approach"] == approach]
        approach_display_name = APPROACH_TO_DISPLAY_NAME[approach]
        color = APPROACH_TO_COLOR[approach]
        sns.lineplot(
            data=approach_df,
            x="Training Execution Time",
            y="Error",
            estimator="mean",
            errorbar="se",
            ax=ax,
            color=color,
            label=approach_display_name,
        )
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=500)
    print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("outfile", type=Path)
    args = parser.parse_args()
    _main(args.results_dir, args.outfile)
