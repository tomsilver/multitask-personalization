"""Hacky FEAST pregeneration for website study."""

from multitask_personalization.feast_dummy import Meal, MEALS, generate_bite_orderings
from multitask_personalization.envs.feeding.feeding_scene_spec import create_feeding_scene_description_from_config
from multitask_personalization.methods.csp_approach import CSPApproach
from multitask_personalization.csp_solvers import RandomWalkCSPSolver
from multitask_personalization.envs.feeding.feeding_structs import  FeedingInitializationQueryObservation, FeedingInitializationDatasetObservation
from tomsutils.llm import OpenAILLM
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

def pregenerate_initialization_variable(var_name: str, model, current_model_state: dict, remaining_meals: list[Meal], outdir: Path, dry_run: bool) -> None:
    global TOTAL_PREDICTIONS
    if not remaining_meals:
        return
    outdir.mkdir(exist_ok=True)
    meal = remaining_meals[0]
    choices = get_choices_for_initialization_variable(var_name, meal)
    bite_ordering_options = get_choices_for_initialization_variable("bite_ordering", meal)
    obs = FeedingInitializationQueryObservation(meal.context, meal.table_type, meal.food_items, meal.dips, bite_ordering_options)
    model.load_from_state(current_model_state)
    if dry_run:
        prediction = "PLACEHOLDER"
    else:
        print("STARTING PREDICTION", TOTAL_PREDICTIONS)
        prediction = model.get_most_preferred_choice(obs)
    TOTAL_PREDICTIONS += 1
    prediction_file = outdir / "prediction.txt"
    with open(prediction_file, "w") as f:
        f.write(prediction)
    metadata = asdict(meal)
    metadata["choices"] = choices
    metadata["llm_model_summary"] = current_model_state["summary_preferences"]
    metadata["history"] = [getattr(o, var_name) for o in current_model_state["data_obs_history"]]
    metadata_file = outdir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)
    for choice in choices:
        choice_outdir = outdir / str(choice)
        feeding_side = choice if var_name == "feeding_side" else None
        bite_ordering = choice if var_name == "bite_ordering" else None
        ready_signal = choice if var_name == "ready_signal" else None
        be_verbal = choice if var_name == "be_verbal" else None
        assert sum([x is not None for x in [feeding_side, bite_ordering, ready_signal, be_verbal]]) == 1
        next_obs = FeedingInitializationDatasetObservation(meal.context, meal.table_type, meal.food_items, meal.dips, bite_ordering_options,
                                                           feeding_side, bite_ordering, ready_signal, be_verbal)
        if not dry_run:
            model.load_from_state(current_model_state)
            model.learn_incremental(next_obs)
            next_model_state = model.get_save_state()
        else:
            next_model_state = current_model_state.copy()
            next_model_state["data_obs_history"] = current_model_state["data_obs_history"] + [next_obs]
        pregenerate_initialization_variable(var_name, model, next_model_state, remaining_meals[1:], choice_outdir, dry_run)

        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--var",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("pregeneration"),
    )
    parser.add_argument(
        "--max_meals",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--dry",
        action="store_true"
    )
    args = parser.parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(exist_ok=True)

    # NOTE: this config shouldn't really matter, we're just accessing the models inside the CSP generator.
    config_file = Path(__file__).parent / "envs" / "feeding" / "configs" / "meal_1.yaml"
    scene_spec = create_feeding_scene_description_from_config(config_file)
    csp_solver = RandomWalkCSPSolver(0)
    llm = OpenAILLM("gpt-4o-mini", Path("feast_llm_cache"))
    explore_method = "exploit-only"
    approach = CSPApproach(scene_spec, None,
                            csp_solver=csp_solver,
                            llm=llm,
                            explore_method=explore_method)
    initialization_var_models = {
        "feeding_side": approach._csp_generator._feeding_side_model,
        "bite_ordering": approach._csp_generator._bite_ordering_model,
        "ready_signal": approach._csp_generator._ready_signal_model,
        "be_verbal": approach._csp_generator._be_verbal_model,
    }

    num_meals = min(len(MEALS), args.max_meals)
    meals = MEALS[:num_meals]

    var_name = args.var
    assert var_name in initialization_var_models
    model = initialization_var_models[var_name]
    init_model_state = model.get_save_state()
    print(f"Running pregeneration for {var_name}")
    pregenerate_initialization_variable(var_name, model, init_model_state, meals, outdir / var_name, dry_run=args.dry)
    print(f"Made {TOTAL_PREDICTIONS} predictions for {var_name}")
