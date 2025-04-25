"""Hacky FEAST pregeneration for website study."""

from multitask_personalization.feast_integration import MultitaskPersonalizationFeastInterface
from multitask_personalization.feast_dummy import Meal, MEALS, generate_bite_orderings
from pathlib import Path
from dataclasses import asdict
import json


def get_choices_for_initialization_variable(var_name: str, meal: Meal) -> list[str]:
    if var_name == "bite_ordering":
        return generate_bite_orderings(meal.food_items, meal.dips)
    if var_name == "feeding_side":
        return ["left", "right"]
    if var_name == "ready_signal":
        return ["mouth_open", "button", "auto_continue"]
    if var_name == "be_verbal":
        return [True, False]
    raise NotImplementedError


TOTAL_PREDICTIONS = 0

def pregenerate_initialization_variable(var_name: str, remaining_meals: list[Meal], outdir: Path) -> None:
    global TOTAL_PREDICTIONS
    if not remaining_meals:
        return
    outdir.mkdir(exist_ok=True)
    meal = remaining_meals[0]
    meal_metadata = asdict(meal)
    metadata_file = outdir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(meal_metadata, f)
    choices = get_choices_for_initialization_variable(var_name, meal)
    prediction = "TODO"
    TOTAL_PREDICTIONS += 1
    prediction_file = outdir / "prediction.txt"
    with open(prediction_file, "w") as f:
        f.write(prediction)
    for choice in choices:
        choice_outdir = outdir / str(choice)
        pregenerate_initialization_variable(var_name, remaining_meals[1:], choice_outdir)

        


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("pregeneration"),
    )
    args = parser.parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(exist_ok=True)

    var_names = ["feeding_side"]
    for var_name in var_names:
        print(f"Running pregeneration for {var_name}")
        pregenerate_initialization_variable(var_name, MEALS, outdir / var_name)
        print(f"Made {TOTAL_PREDICTIONS} predictions for {var_name}")
