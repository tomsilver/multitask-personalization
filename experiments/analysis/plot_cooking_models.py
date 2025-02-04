"""Create animated plots for cooking model learning."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from omegaconf import DictConfig, OmegaConf

from multitask_personalization.utils import Bounded1DClassifier


def _main(
    results_dir: Path, outfile: Path, meal_name: str, ingredient_name: str, fps: int = 2
) -> None:
    # Load the learned model data.
    data: dict = {
        n: {
            "step": [],
            "x1": [],
            "x2": [],
            "x3": [],
            "x4": [],
            "incremental_X": [],
            "incremental_Y": [],
        }
        for n in ["temperature", "quantity"]
    }

    model_dir = results_dir / "models"
    for checkpoint_dir in sorted(
        [d for d in model_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    ):
        num_training_steps = int(checkpoint_dir.name)
        model_file = checkpoint_dir / "meal_preferences.json"
        with open(model_file, "r", encoding="utf-8") as f:
            model_params = json.load(f)[meal_name][ingredient_name]
        for name, d in data.items():
            d["step"].append(num_training_steps)
            for k in model_params[name]:
                d[k].append(model_params[name][k])

    # Load the ground truth parameters.
    config_path = results_dir / "config.yaml"
    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)
    meal_specs = {
        m.name: m for m in cfg.env.env.hidden_spec.meal_preference_model.meal_specs
    }
    meal_spec = meal_specs[meal_name]
    ing_specs = {i.name: i for i in meal_spec.ingredients}
    ground_truth_spec = ing_specs[ingredient_name]

    # Set up the figure and axis.
    fig, axes = plt.subplots(1, len(data), sharey=True)
    assert isinstance(axes, np.ndarray)
    fig.suptitle(f"{meal_name}: {ingredient_name}")
    artists: list = []
    for i, metric_name in enumerate(sorted(data)):
        axes[i].set_xlabel(metric_name)
        if i == 0:
            axes[i].set_ylabel("P(True)")
        min_x = min(data[metric_name]["incremental_X"][-1])
        max_x = max(data[metric_name]["incremental_X"][-1])
        pad = (max_x - min_x) * 0.1
        axes[i].set_xlim((min_x - pad, max_x + pad))
        axes[i].set_ylim((-0.1, 1.1))
        axes[i].vlines(
            ground_truth_spec[metric_name], -0.1, 1.1, linestyle="--", color="green"
        )
    plt.tight_layout()

    # Define the initialization function.
    def init():
        return artists

    # Define the animation function.
    def animate(t):
        while artists:
            artist = artists.pop()
            artist.remove()
        for i, metric_name in enumerate(sorted(data)):
            x = data[metric_name]["incremental_X"][t]
            y = data[metric_name]["incremental_Y"][t]
            scatter_plot = axes[i].scatter(x, y, color="black")
            artists.append(scatter_plot)
            x_stars = [data[metric_name][xs][t] for xs in ["x1", "x2", "x3", "x4"]]
            y_stars = [0, 1, 1, 0]
            scatter_plot = axes[i].scatter(
                x_stars, y_stars, color="gold", marker="*", s=250
            )
            artists.append(scatter_plot)
            model = Bounded1DClassifier(-100, 100)  # too lazy to load bounds
            model.x1 = x_stars[0]
            model.x2 = x_stars[1]
            model.x3 = x_stars[2]
            model.x4 = x_stars[3]
            model.incremental_X = x
            model.incremental_Y = y
            x_min, x_max = axes[i].get_xlim()
            X_test = np.linspace(x_min, x_max, num=100, endpoint=True)
            Y_test = model.predict_proba(X_test)
            (line_plot,) = axes[i].plot(X_test, Y_test, color="red")
            artists.append(line_plot)
        return []

    # Create the animation.
    ani = animation.FuncAnimation(
        fig, animate, frames=len(data["temperature"]["step"]), init_func=init, blit=True
    )

    # Save the animation.
    ani.save(outfile, writer="ffmpeg", fps=fps)
    print(f"Wrote out to {outfile}")

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("outfile", type=Path)
    parser.add_argument("--meal_name", type=str, default="seasoning")
    parser.add_argument("--ingredient_name", type=str, default="salt")
    args = parser.parse_args()
    _main(args.results_dir, args.outfile, args.meal_name, args.ingredient_name)
