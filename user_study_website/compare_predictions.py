#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_responses(file_path):
    """
    Load Google Form responses from a text file where each line is a JSON string.
    
    Args:
        file_path: Path to the text file containing responses
        
    Returns:
        List of parsed JSON responses
    """
    responses = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    response = json.loads(line)
                    responses.append(response)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line as JSON: {line[:50]}...")
    
    return responses


def _occlusion_answer_to_string(occlusion):
    relevant_poi_choice_str = "none" if not occlusion["relevant_pois"] else "-".join(occlusion["relevant_pois"])
    occluded_poi_choice_str = "none" if not occlusion["occluded_pois"] else "-".join( occlusion["occluded_pois"])
    choice_str = relevant_poi_choice_str + "___" + occluded_poi_choice_str
    return choice_str


def _get_simplified_responses(responses):
    responses = load_responses(file_path)
    simplified_responses = []
    for response in responses:
        simplified_response = []
        for answer in response['answers']:
            reduced_answer = {
                'occlusion': _occlusion_answer_to_string(answer['occlusion']),
                'bite_order': answer['bite_order']['value'],
                'ready_signal': answer['ready_signal']['value'],
                'verbal': eval(answer['verbal']['value']),
            }
            simplified_response.append(reduced_answer)
        simplified_responses.append(simplified_response)
    return simplified_responses


def _get_model_prediction(variable, current_path):
    if variable == 'occlusion':
        with open(current_path / "prediction.json", 'r') as f:
            prediction = json.load(f)
        prediction_str = _occlusion_answer_to_string(prediction)
        return prediction_str
    if variable == 'bite_order':
        with open(current_path / "prediction.txt", 'r') as f:
            prediction = f.read().strip()
        return prediction
    if variable == 'ready_signal':
        with open(current_path / "prediction.txt", 'r') as f:
            prediction = f.read().strip()
        return prediction
    if variable == 'verbal':
        with open(current_path / "prediction.txt", 'r') as f:
            prediction = f.read().strip()
        return eval(prediction)
    raise NotImplementedError


def _get_non_personalized_prediction(variable, meal_id):
    if variable == "occlusion":
        # For occlusion, we use a fixed prediction
        return "none___none"
    path = Path(__file__).parent / "content" / "non_personalized" / _variable_to_outer_dir(variable) / f"meal{meal_id}"
    return _get_model_prediction(variable, path)


def _variable_to_outer_dir(variable):
    if variable == "bite_order":
        outer_dir = "bite_ordering"
    elif variable == "verbal":
        outer_dir = "be_verbal"
    else:
        outer_dir = variable
    return outer_dir


def _get_prediction_success_sequence(variable, var_sequence, non_personalized):
    outer_dir = _variable_to_outer_dir(variable)

    current_path = Path(__file__).parent / "content" / outer_dir
    assert current_path.exists(), f"Path {current_path} does not exist."
    success_sequence = []
    for i, user_selected_value in enumerate(var_sequence):
        meal_id = i + 1
        if non_personalized:
            model_prediction = _get_non_personalized_prediction(variable, meal_id)
        else:
            model_prediction = _get_model_prediction(variable, current_path)
        success = 1 if user_selected_value == model_prediction else 0
        success_sequence.append(success)
        current_path = current_path / str(user_selected_value)
    return success_sequence


def create_results_dataframe(var_to_prediction_success_sequences):
    """
    Create a DataFrame from the prediction success sequences.
    
    Args:
        var_to_prediction_success_sequences: Dictionary mapping variables to their prediction success sequences
        
    Returns:
        DataFrame with columns:
        - prediction_type: The type of prediction (occlusion, bite_order, ready_signal, or verbal)
        - meal_number: The meal number (1-based index)
        - participant_id: The index of the participant
        - success: Whether the prediction was successful (0 or 1)
    """
    rows = []
    for variable, success_sequences in var_to_prediction_success_sequences.items():
        # Convert to numpy array for easier calculation
        success_sequences = np.array(success_sequences)
        # For each participant and meal
        for participant_id in range(success_sequences.shape[0]):
            for meal_number in range(success_sequences.shape[1]):
                success = success_sequences[participant_id, meal_number]
                rows.append({
                    'prediction_type': variable,
                    'meal_number': meal_number + 1,
                    'participant_id': participant_id,
                    'success': success
                })
    return pd.DataFrame(rows)


def main(file_path, non_personalized):
    responses = load_responses(file_path)
    simplified_responses = _get_simplified_responses(responses)
    variables = ['occlusion', 'bite_order', 'ready_signal', 'verbal']
    var_to_prediction_success_sequences = {}
    for variable in variables:
        print(f"Variable: {variable}")
        var_prediction_success_sequences = []
        for response in simplified_responses:
            var_sequence = [answer[variable] for answer in response]
            success_sequence = _get_prediction_success_sequence(variable, var_sequence, non_personalized)
            var_prediction_success_sequences.append(success_sequence)
        var_to_prediction_success_sequences[variable] = var_prediction_success_sequences
    
    # Create DataFrame and save to CSV
    df = create_results_dataframe(var_to_prediction_success_sequences)
    output_file = f"raw_predictions{'__non_personalized' if non_personalized else ''}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nRaw prediction data saved to {output_file}")
    
    # Print summary statistics
    for variable, success_sequences in var_to_prediction_success_sequences.items():
        print(f"Variable: {variable}")
        avg_success_sequence = np.mean(success_sequences, axis=0)
        print(f"Average prediction success sequence: {avg_success_sequence}")


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Analyze meal preference data from Google Form responses.')
    parser.add_argument('file_path', nargs='?', default='form_responses.txt',
                        help='Path to the form responses file (default: form_responses.txt)')
    parser.add_argument('--non_personalized', action='store_true')

    # Parse arguments
    args = parser.parse_args()
    file_path = Path(args.file_path)
    
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist.")

    else:
        main(file_path, args.non_personalized)
