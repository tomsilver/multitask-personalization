"""Plot prediction error over time for learned ROM models."""

import argparse
import json
import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from plot_main_results import (
    APPROACH_TO_COLOR,
    APPROACH_TO_DISPLAY_NAME,
    ENV_TO_DISPLAY_NAME,
)

from multitask_personalization.rom.models import ROMModel, SphericalROMModel
from multitask_personalization.utils import (
    sample_within_sphere,
)


def _calculate_prediction_error(
    checkpoint_dir: Path, cfg: DictConfig, eval_data: tuple[list[NDArray], list[bool]]
) -> float:
    model = hydra.utils.instantiate(cfg.rom_model)
    assert isinstance(model, ROMModel)
    model.load(checkpoint_dir)
    predictions: list[bool] = []
    positions, labels = eval_data
    for position in positions:
        prediction = model.check_position_reachable(position)
        predictions.append(prediction)
    error = sum(np.not_equal(predictions, labels)) / len(labels)
    return error


def _create_eval_data(
    ground_truth_rom_model: ROMModel,
    seed: int,
    num_samples: int = 100,
    min_radius: float = 0.0,
    max_radius: float = 1.5,
    balance_data: bool = True,
) -> tuple[list[NDArray], list[bool]]:
    positions: list[NDArray] = []
    labels: list[bool] = []
    assert isinstance(ground_truth_rom_model, SphericalROMModel)
    sphere_center = (
        ground_truth_rom_model._sphere_center
    )  # pylint: disable=protected-access
    rng = np.random.default_rng(seed)
    while len(positions) < num_samples:
        position = np.array(
            sample_within_sphere(sphere_center, min_radius, max_radius, rng)
        )
        label = ground_truth_rom_model.check_position_reachable(position)
        if balance_data:
            if label and (sum(labels) >= num_samples // 2):
                continue
            if not label and (len(labels) - sum(labels) >= num_samples // 2):
                continue
        positions.append(position)
        labels.append(label)
    return positions, labels


def _main(
    results_dir: Path, outfile: Path, config_filename: str = "config.yaml"
) -> None:
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
            # Create ground-truth model from this config.
            rom_model = hydra.utils.instantiate(cfg.rom_model)
            # Create eval data.
            eval_data = _create_eval_data(rom_model, seed)
            for checkpoint_dir in sorted(
                [d for d in model_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                key=lambda d: int(d.name),
            ):
                num_training_steps = int(checkpoint_dir.name)
                training_time = num_training_steps * cfg.env.dt
                error = _calculate_prediction_error(checkpoint_dir, cfg, eval_data)
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
