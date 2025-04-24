import rospy
import pickle
import base64
import time
import itertools
from dataclasses import dataclass
from typing import List
from std_msgs.msg import String
import argparse
from pybullet_helpers.geometry import Pose
import numpy as np


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Feast Dummy")
    parser.add_argument("--meal_id", type=int, default=0, help="ID of the meal to use (0-4)")
    args = parser.parse_args()

    rospy.init_node("feast_dummy", anonymous=True)

    # Helper function to send a request to the multitask personalization module.
    def _send_mp_request(data):
        # Encode the message.
        global _mp_response
        s = pickle.dumps(data)
        s = base64.b64encode(s).decode('ascii')
        msg = String()
        msg.data = s
        _mp_response = None
        mp_request_pub.publish(msg)
        print("Sent MP request: ", data)  
        while _mp_response is None:
            time.sleep(0.1)  # Wait for the response
        print("Received MP response: ", _mp_response)
        return _mp_response

    # Response callback function
    def mp_response_callback(msg):
        # Decode the message.
        s = base64.b64decode(msg.data.encode('ascii'))
        data = pickle.loads(s)
        global _mp_response
        _mp_response = data

    mp_response_sub = rospy.Subscriber("/mp_response", String, mp_response_callback)
    mp_request_pub = rospy.Publisher("/mp_request", String, queue_size=10)
    time.sleep(1)  # Wait for the subscriber and publisher to be ready
    
    # Helper function to verify predictions with the user.
    def verify_predictions(prediction, options):
        if prediction not in options:
            raise ValueError(f"Invalid prediction: {prediction}. Expected one of {options}.")
        print("From the following options:")
        for i in range(len(options)):
            print(f"{i+1}. {options[i]}")
        print("The robot predicted the following preference: ", prediction)
        user_input = input("Do you agree with the robot's prediction? (y/n): ")
        while user_input not in ["y", "n"]:
            user_input = input("Please enter 'y' or 'n': ")
        if user_input == "y":
            print("User agreed with the robot's prediction.")
            return prediction
        else:
            # get user's preference
            preferred_id = input("Please enter the number of your preferred option: ")
            while not preferred_id.isdigit() or int(preferred_id) < 1 or int(preferred_id) > len(options):
                preferred_id = input("Please enter a valid number: ")
            preferred_id = int(preferred_id) - 1
            print(f"User preferred option: {options[preferred_id]}")
            return options[preferred_id]
        
    # Helper function to generate all possible bite orderings.
    def generate_bite_orderings(food_items: List[str], dips: List[str]) -> List[str]:
        orderings = []

        # All permutations of food items with all dipping combinations
        for food_perm in itertools.permutations(food_items):
            food_dip_variants = []
            for food in food_perm:
                variants = [f"{food} without any dipping"]
                variants += [f"{food} dipped in {dip}" for dip in dips]
                food_dip_variants.append(variants)

            for combo in itertools.product(*food_dip_variants):
                orderings.append("then ".join(combo))

        if len(food_items) > 1:
            # One alternating pattern across all food items
            alt_variants = []
            for food in food_items:
                if dips:
                    alt_variants.append(f"{food} dipped in {dips[0]}")
                else:
                    alt_variants.append(food)
            if len(alt_variants) == 1:
                alt_pattern = f"alternating bites of {alt_variants[0]}"
            else:
                alt_pattern = "alternating bites of " + " and ".join(alt_variants)
            orderings.append(alt_pattern)

        return orderings
        
    @dataclass
    class Meal:
        context: str
        table_type: str
        food_items: List[str]
        dips: List[str]

    meals = [
        Meal("personal", "rectangular table", ["chicken breast"], ["ketchup", "ranch dressing"]),
        Meal("social with friend on left", "rectangular table", ["celery", "apple slices"], ["ranch dressing"]),
        Meal("TV-watching", "circular table", ["steak", "potatoes"], []),
        Meal("personal", "rectangular table", ["celery", "pear slices"], ["ranch dressing"]),
        Meal("social TV-watching with friend on right side", "circular table", ["chicken nuggets"], ["ketchup", "ranch dressing"]),
    ]

    current_meal = meals[args.meal_id]
    bite_ordering_options = generate_bite_orderings(current_meal.food_items, current_meal.dips)

    # send mealContext, table_type, food_items and bite_ordering_options to multitask personalization
    mp_response = _send_mp_request({"request_type": "initialization_query",
                        "context": current_meal.context, 
                        "table_type": current_meal.table_type,
                        "food_items": current_meal.food_items,
                        "dips": current_meal.dips,
                        "bite_ordering_options": bite_ordering_options})
    assert mp_response["response_type"] == "initialization_query"
    feeding_side = mp_response["feeding_side"]
    bite_ordering = mp_response["bite_ordering"]
    ready_signal = mp_response["ready_signal"]
    be_verbal = mp_response["be_verbal"]

    # verify predictions with the user (using the terminal)
    feeding_side = verify_predictions(feeding_side, ["left", "right"])
    bite_ordering = verify_predictions(bite_ordering, bite_ordering_options)
    ready_signal = verify_predictions(ready_signal, ["mouth_open", "button", "auto_continue"])
    be_verbal = verify_predictions(be_verbal, [True, False])

    # send the verified predictions to multitask personalization
    mp_response = _send_mp_request({"request_type": "initialization_dataset",
                        "feeding_side": feeding_side, 
                        "bite_ordering": bite_ordering, 
                        "ready_signal": ready_signal,
                        "be_verbal": be_verbal})
    assert mp_response["response_type"] == "initialization_dataset"

    # send plate and drink pose to multitask personalization
    current_plate_pose = Pose((0.4, 0.3, 0.17))
    current_drink_pose = Pose((0.55, 0.6, 0.35), (0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0))
    mp_response = _send_mp_request({"request_type": "occlusion_query",
                                    "plate_pose": current_plate_pose,
                                    "drink_pose": current_drink_pose,
                                    })
    assert mp_response["response_type"] == "occlusion_query"
    plate_delta_xy = mp_response["plate_delta_xy"]
    drink_delta_xy = mp_response["drink_delta_xy"]
    before_transfer_pose = mp_response["before_transfer_pose"]
    before_transfer_pos = mp_response["before_transfer_pos"]
    above_plate_pos = mp_response["above_plate_pos"]
    occlusion_poi_relevance = mp_response["occlusion_poi_relevance"]

    # TODO visualize the potential occlusion points
    new_plate_pose = Pose((current_plate_pose.position[0] + plate_delta_xy[0],
                           current_plate_pose.position[1] + plate_delta_xy[1],
                           current_plate_pose.position[2]),
                           current_plate_pose.orientation)
    new_drink_pose = Pose((current_drink_pose.position[0] + drink_delta_xy[0],
                           current_drink_pose.position[1] + drink_delta_xy[1],
                           current_drink_pose.position[2]),
                           current_drink_pose.orientation)
    occlusion_dataset_dict = {
        "request_type": "occlusion_dataset",
        "plate_pose": new_plate_pose,
        "drink_pose": new_drink_pose,
        "occlusion": {}
    }
    for poi, prediction in occlusion_poi_relevance.items():
        print(f"Verifying the RELEVANCE of POI={poi} for this meal")
        relevance = verify_predictions(prediction, [True, False])
        if relevance:
            print(f"Verifying whether view was occluded for POI={poi} during FEEDING")
            plate_occlusion = verify_predictions(False, [True, False])
            print(f"Verifying whether view was occluded for POI={poi} during DRINKING")
            drink_occlusion = verify_predictions(False, [True, False])
        else:
            plate_occlusion = False
            drink_occlusion = False
        occlusion_dataset_dict["occlusion"][poi] = {
            "relevance": relevance,
            "plate_occlusion": plate_occlusion,
            "drink_occlusion": drink_occlusion,
        }
    mp_response = _send_mp_request(occlusion_dataset_dict)
    
    # TODO: we need to resolve the issue that drink_post_grasp_pose is used to check drink occlusions
    # but it's not actually used on the robot. I took this out earlier because it was causing strange
    # motions on the robot but we need to add it back for consistency. Otherwise occlusion learning
    # won't work if we try to learn from the drink.
