"""Hacky FEAST pregeneration for website study.

Useful commands:
    python src/multitask_personalization/pregeneration.py --var occlusion --test "front___front/front-left___front/front___none/front___none/"
"""

from multitask_personalization.feast_dummy import Meal, MEALS, generate_bite_orderings
from multitask_personalization.envs.feeding.feeding_scene_spec import create_feeding_scene_description_from_config
from multitask_personalization.methods.csp_approach import CSPApproach
from multitask_personalization.csp_solvers import RandomWalkCSPSolver
from pybullet_helpers.geometry import Pose, set_pose
from pybullet_helpers.inverse_kinematics import set_robot_joints_with_held_object
from multitask_personalization.envs.feeding.feeding_structs import  FeedingInitializationQueryObservation, FeedingInitializationDatasetObservation, FeedingOcclusionDatasetObservation, FeedingOcclusionQueryObservation
from multitask_personalization.envs.feeding.feeding_env import FeedingEnv, BANISH_POSE
import itertools
from tomsutils.llm import OpenAILLM
from pathlib import Path
from dataclasses import asdict
import json
import imageio.v2 as iio
import numpy as np
from functools import lru_cache


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

def pregenerate_initialization_variable(var_name: str, model, current_model_state: dict, remaining_meals: list[Meal], outdir: Path, dry_run: bool, nonpersonalized: bool = False, prune_fn=None) -> None:
    global TOTAL_PREDICTIONS
    if not remaining_meals:
        return
    if prune_fn is not None and prune_fn(outdir):
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
        if var_name == "bite_ordering":
            assert isinstance(prediction, int)
            prediction = choices[prediction]
    TOTAL_PREDICTIONS += 1
    prediction_file = outdir / "prediction.txt"
    with open(prediction_file, "w") as f:
        f.write(str(prediction))
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
        if not (dry_run or nonpersonalized):
            model.load_from_state(current_model_state)
            model.learn_incremental(next_obs)
            next_model_state = model.get_save_state()
        else:
            next_model_state = current_model_state.copy()
            next_model_state["data_obs_history"] = current_model_state["data_obs_history"] + [next_obs]
        pregenerate_initialization_variable(var_name, model, next_model_state, remaining_meals[1:], choice_outdir, dry_run, nonpersonalized, prune_fn=prune_fn)

    

def get_choices_for_occlusion_poi(meal: Meal) -> list[str]:
    if meal.meal_id in [1, 3, 4]:
        return ["front"]
    assert meal.meal_id in [2, 5]
    return ["front", "left"]


def get_all_subsets(lst):
    subsets = []
    for r in range(len(lst)+1):
        subsets.extend(itertools.combinations(lst, r))
    return subsets

def render_bite_occlusion_image(sim: FeedingEnv, plate_pose: Pose, drink_pose: Pose, robot_joints: list[float]) -> None:
    # Set the plate and drink poses in the simulation.
    set_pose(sim.get_object_id_from_name("plate"), plate_pose, sim.physics_client_id)
    set_pose(sim.get_object_id_from_name("drink"), drink_pose, sim.physics_client_id)

    held_object_id = sim.get_object_id_from_name("utensil")
    held_object_tf = sim.scene_spec.utensil_held_object_tf
    set_robot_joints_with_held_object(
        sim.robot,
        sim.physics_client_id,
        held_object_id,
        held_object_tf,
        robot_joints + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    sim.robot.set_finger_state(
        sim.scene_spec.tool_grasp_fingers_value
    )

    # Render the image.
    img = sim.render(user_view=True)
    return img


@lru_cache(maxsize=None)
def get_sim(meal_id):
    config_file = Path(__file__).parent / "envs" / "feeding" / "configs" / f"meal_{meal_id}.yaml"
    scene_spec = create_feeding_scene_description_from_config(config_file)
    return FeedingEnv(scene_spec)


def pregenerate_occlusion(approach: CSPApproach, current_approach_state, init_plate_pose: Pose, remaining_meals: list[Meal], outdir: Path, dry_run: bool, prune_fn=None) -> None:
    global TOTAL_PREDICTIONS
    if not remaining_meals:
        return
    if prune_fn is not None and prune_fn(outdir):
        return
    outdir.mkdir(exist_ok=True)
    meal = remaining_meals[0]
    possible_pois = get_choices_for_occlusion_poi(meal)
    approach.load_from_state(current_approach_state)

    llm_models = approach._csp_generator._occlusion_poi_relevance_models
    occlusion_model = approach._csp_generator._occlusion_model

    bite_ordering_options = get_choices_for_initialization_variable("bite_ordering", meal)

    # Predict the next plate pose using the current model.
    approach._csp_generator._disable_drink = True
    obs = FeedingOcclusionQueryObservation(meal.context, meal.table_type, meal.food_items, meal.dips, bite_ordering_options, init_plate_pose, BANISH_POSE)
    approach.reset(obs, {})
    try:
        act = approach.step()
    except:
        import ipdb; ipdb.set_trace()
    plate_pose = Pose((init_plate_pose.position[0] + act.plate_delta_xy[0],
                            init_plate_pose.position[1] + act.plate_delta_xy[1],
                            init_plate_pose.position[2]),
                            init_plate_pose.orientation)
    plate_position = plate_pose.position[:2]
    if not dry_run:
        viz_sim = get_sim(meal.meal_id)
        bite_occlusion_image =  render_bite_occlusion_image(viz_sim, plate_pose, BANISH_POSE, act.above_plate_pos)
        occlusion_img_outfile = outdir / "bite_occlusion_image.png"
        iio.imsave(occlusion_img_outfile, bite_occlusion_image)
    sim = approach._csp_generator._sim
    set_pose(sim.get_object_id_from_name("plate"), plate_pose, sim.physics_client_id)
    
    # Predict which of the subset of points of interest are relevant.
    relevant_pois = set()
    for poi in possible_pois:
        model = llm_models[poi]
        if dry_run:
            prediction = True
        else:
            print("STARTING PREDICTION", TOTAL_PREDICTIONS)
            prediction = model.get_most_preferred_choice(obs)
        TOTAL_PREDICTIONS += 1
        if prediction:
            relevant_pois.add(poi)
    # Predict which of the predicted-relevant points of interest have occlusions for the plate.
    occlusion_scale = (
        1.0 - (occlusion_model.post_max + occlusion_model.post_min) / 2
    )
    occluded_pois = set()
    for poi in relevant_pois:
        score = approach._csp_generator._get_plate_occlusion_score(plate_position, poi)
        assert score is not None
        occluded = score >= 1.0 - occlusion_scale
        if occluded:
            occluded_pois.add(poi)
    prediction_file = outdir / "prediction.json"
    saved_prediction = {
        "relevant_pois": sorted(relevant_pois),
        "occluded_pois": sorted(occluded_pois),
    }
    with open(prediction_file, "w") as f:
        json.dump(saved_prediction, f)
    metadata = asdict(meal)
    metadata["choices"] = possible_pois
    metadata["occlusion_scale"] = occlusion_scale
    metadata_file = outdir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)
    # For each possible set of occlusion reports that the study participant might choose, branch.
    # NOTE: it is important to separately ask about relevance and occlusions, otherwise supervision
    # will get messed up.
    for relevant_poi_choice in get_all_subsets(sorted(possible_pois)):
        relevant_poi_choice_str = "none" if not relevant_poi_choice else "-".join(relevant_poi_choice)
        for occluded_poi_choice in get_all_subsets(relevant_poi_choice):
            occluded_poi_choice_str = "none" if not occluded_poi_choice else "-".join(occluded_poi_choice)
            choice_str = relevant_poi_choice_str + "___" + occluded_poi_choice_str
            choice_outdir = outdir / choice_str
            if prune_fn is not None and prune_fn(choice_outdir):
                continue

            occlusion_dict = {}
            for poi in possible_pois:
                occlusion_dict[poi] = {
                    "relevance": (poi in relevant_poi_choice),
                    "plate_occlusion": (poi in occluded_poi_choice),
                    "drink_occlusion": False,
                }
            next_obs = FeedingOcclusionDatasetObservation(meal.context, meal.table_type, meal.food_items, meal.dips, bite_ordering_options,
                                                          plate_pose, BANISH_POSE, occlusion_dict)
            if not dry_run:
                approach.load_from_state(current_approach_state)
                # Update occlusion relevance models.
                approach._csp_generator.observe_transition(obs, act, next_obs, False, {})
                next_approach_state = approach.get_save_state()
            else:
                next_approach_state = current_approach_state.copy()
            pregenerate_occlusion(approach, next_approach_state, plate_pose, remaining_meals[1:], choice_outdir, dry_run, prune_fn)



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
    parser.add_argument(
        "--test",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--non_personalized",
        action="store_true"
    )
    parser.add_argument(
        "--use_gui",
        action="store_true"
    )
    args = parser.parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(exist_ok=True)
    test_name = args.test or None

    # NOTE: this config shouldn't really matter, we're just accessing the models inside the CSP generator.
    config_file = Path(__file__).parent / "envs" / "feeding" / "configs" / "meal_1.yaml"
    scene_spec = create_feeding_scene_description_from_config(config_file)
    csp_solver = RandomWalkCSPSolver(0)
    llm = OpenAILLM("gpt-4o-mini", Path("feast_llm_cache"))
    explore_method = "exploit-only"
    approach = CSPApproach(scene_spec, None,
                            csp_solver=csp_solver,
                            llm=llm,
                            explore_method=explore_method,
                            use_gui=args.use_gui)
    current_approach_state = approach.get_save_state()
    initialization_var_models = {
        "feeding_side": approach._csp_generator._feeding_side_model,
        "bite_ordering": approach._csp_generator._bite_ordering_model,
        "ready_signal": approach._csp_generator._ready_signal_model,
        "be_verbal": approach._csp_generator._be_verbal_model,
    }

    num_meals = min(len(MEALS), args.max_meals)
    meals = MEALS[:num_meals]

    var_name = args.var
    if test_name is not None:
        target_dir = str(outdir / var_name / test_name) + "/"
        prune_fn = lambda o: not target_dir.startswith(str(o) + "/")
    else:
        prune_fn = None

    if args.non_personalized:
        assert var_name in initialization_var_models
        for meal in MEALS:
            model = initialization_var_models[var_name]
            init_model_state = model.get_save_state()
            print(f"Running pregeneration for {var_name}")
            subdir = outdir / "non_personalized" / var_name
            subdir.mkdir(parents=True, exist_ok=True)
            pregenerate_initialization_variable(var_name, model, init_model_state, [meal], subdir / f"meal{meal.meal_id}", dry_run=args.dry, nonpersonalized=True, prune_fn=prune_fn)
            model.load_from_state(init_model_state)
            print(f"Made {TOTAL_PREDICTIONS} predictions for {var_name}")

    else:

        if var_name in initialization_var_models:
            model = initialization_var_models[var_name]
            init_model_state = model.get_save_state()
            print(f"Running pregeneration for {var_name}")
            pregenerate_initialization_variable(var_name, model, init_model_state, meals, outdir / var_name, dry_run=args.dry, prune_fn=prune_fn)
            print(f"Made {TOTAL_PREDICTIONS} predictions for {var_name}")

        else:
            assert var_name == "occlusion"
            plate_pose = Pose((0.3, 0.75, 0.17))
            print(f"Running pregeneration for {var_name}")
            approach_state = approach.get_save_state()
            pregenerate_occlusion(approach, approach_state, plate_pose, meals, outdir / var_name, dry_run=args.dry,
                                prune_fn=prune_fn)
            print(f"Made {TOTAL_PREDICTIONS} predictions for {var_name}")
