"""Create animated plots for cooking model learning."""

import argparse
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import json
import matplotlib.pyplot as plt
from multitask_personalization.utils import Bounded1DClassifier
from matplotlib import animation


def _main(results_dir: Path, outfile: Path, meal_name: str, ingredient_name: str,
           fps: int = 2) -> None:
    # Load the learned model data.
    data = {
        n: {
        "step": [],
        "x1": [],
        "x2": [],
        "x3": [],
        "x4": [],
        "incremental_X": [],
        "incremental_Y": [],
    } for n in ["temperature", "quantity"]
    }

    model_dir = results_dir / "models"
    for checkpoint_dir in sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                           key=lambda d: int(d.name)):
        num_training_steps =  int(checkpoint_dir.name)
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
    meal_specs = {m.name: m for m in cfg.env.env.hidden_spec.meal_preference_model.meal_specs}
    meal_spec = meal_specs[meal_name]
    ing_specs = {i.name: i for i in meal_spec.ingredients}
    ing_spec = ing_specs[ingredient_name]
    ground_truth_temperature = ing_spec.temperature
    ground_truth_quantity = ing_spec.quantity

    # Set up the figure and axis.
    fig, axes = plt.subplots(1, len(data), sharey=True)
    fig.suptitle(f"{meal_name}: {ingredient_name}")
    scatter_plots = []
    for i, metric_name in enumerate(sorted(data)):
        axes[i].set_xlabel(metric_name)
        if i == 0:
            axes[i].set_ylabel("P(True)")
        min_x = min(data[metric_name]["incremental_X"][-1])
        max_x = max(data[metric_name]["incremental_X"][-1])
        axes[i].set_xlim((min_x - 0.1, max_x + 0.1))
        axes[i].set_ylim((-0.1, 1.1))
        # Initialize an empty scatter plot.
        scatter_plot = axes[i].scatter([], [], label="data")
        scatter_plots.append(scatter_plot)
    plt.tight_layout()

    # Define the initialization function.
    def init():
        return scatter_plots
    
    # Define the animation function.
    def animate(t):
        while scatter_plots:
            scatter_plot = scatter_plots.pop()
            scatter_plot.remove()
        for i, metric_name in enumerate(sorted(data)):
            x = data[metric_name]["incremental_X"][t]
            y = data[metric_name]["incremental_Y"][t]
            scatter_plot = axes[i].scatter(x, y, label="data", color="black")
            scatter_plots.append(scatter_plot)
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
