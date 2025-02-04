"""Create animated plots for cooking model learning."""

import argparse
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import json


def _main(results_dir: Path, outfile: Path, meal_name: str, ingredient_name: str) -> None:
    data = {
        n: {
        "training_steps": [],
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
            d["training_steps"].append(num_training_steps)
            for k in model_params[name]:
                d[k] = model_params[name][k]

    config_path = results_dir / "config.yaml"
    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)
    meal_specs = {m.name: m for m in cfg.env.env.hidden_spec.meal_preference_model.meal_specs}
    meal_spec = meal_specs[meal_name]
    ing_specs = {i.name: i for i in meal_spec.ingredients}
    ing_spec = ing_specs[ingredient_name]
    ground_truth_temperature = ing_spec.temperature
    ground_truth_quantity = ing_spec.quantity
    import ipdb; ipdb.set_trace()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("outfile", type=Path)
    parser.add_argument("--meal_name", type=str, default="seasoning")
    parser.add_argument("--ingredient_name", type=str, default="salt")
    args = parser.parse_args()
    _main(args.results_dir, args.outfile, args.meal_name, args.ingredient_name)
