import pandas as pd
import json
import base64
import os

def decode_response(encoded_string):
    """
    Decode a base64 encoded Google form response string into a Python dictionary.
    
    Args:
        encoded_string (str): Base64 encoded response string
        
    Returns:
        dict: Decoded response data
    """
    try:
        # Decode the base64 string
        decoded_bytes = base64.b64decode(encoded_string.strip())
        decoded_json = decoded_bytes.decode('utf-8')
        
        # Parse the JSON
        data = json.loads(decoded_json)
        
        # Return the extracted data
        return data
    except Exception as e:
        print(f"Error decoding response: {e}")
        return None

def process_file(filepath):
    """
    Process a text file where each line is a base64 encoded Google form response.
    
    Args:
        filepath (str): Path to the text file
        
    Returns:
        pd.DataFrame: DataFrame containing the decoded responses
    """
    # List to store all processed responses
    all_responses = []
    
    # Read the file line by line
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue  # Skip empty lines
                
            # Decode the response
            response = decode_response(line)
            
            if response:
                # Flatten the response structure
                flattened = flatten_response(response, i+1)
                all_responses.append(flattened)
    
    # Convert to DataFrame
    if all_responses:
        df = pd.DataFrame(all_responses)
        return df
    else:
        print("No valid responses found in the file.")
        return pd.DataFrame()

def flatten_response(response, response_id):
    """
    Flatten the nested response structure for easier analysis.
    
    Args:
        response (dict): Decoded response dictionary
        response_id (int): Response identifier
        
    Returns:
        dict: Flattened response
    """
    flattened = {'response_id': response_id}
    
    # Extract answers and intake data
    if 'answers' in response:
        answers = response['answers']
        
        # Add flattened participant info
        if 'intakeData' in response:
            intake = response['intakeData']
            if intake:
                for key, value in intake.items():
                    flattened[f'participant_{key}'] = value
        
        # Process each meal's answers
        for meal_idx, meal_answer in enumerate(answers):
            meal_num = meal_idx + 1
            
            # Add preference rating for this meal
            if 'preference_rating' in meal_answer:
                flattened[f'meal{meal_num}_preference'] = meal_answer['preference_rating']
            
            # Add which option was personalized
            if 'isOptionAPersonalized' in meal_answer:
                flattened[f'meal{meal_num}_optionA_personalized'] = meal_answer['isOptionAPersonalized']
            
            # Add other question answers
            for question_key in ['bite_order', 'ready_signal', 'verbal', 
                                'look_forward', 'block_forward', 
                                'look_left', 'block_left']:
                if question_key in meal_answer:
                    value = meal_answer[question_key]['value'] if isinstance(meal_answer[question_key], dict) else meal_answer[question_key]
                    flattened[f'meal{meal_num}_{question_key}'] = value
    
    return flattened

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python decode_responses.py <responses_file.txt>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        # Process the file and create DataFrame
        df = process_file(file_path)
        
        # Print the DataFrame
        if not df.empty:
            print(f"Successfully decoded {len(df)} responses.")
            print("\nDataFrame Preview:")
            print(df)
            
            # Print column names for reference
            print("\nAvailable columns:")
            for col in df.columns:
                print(f"- {col}")
                
            # Save to CSV
            output_file = os.path.splitext(file_path)[0] + '.csv'
            df.to_csv(output_file, index=False)
            print(f"\nDataFrame saved to {output_file}")
    except Exception as e:
        print(f"Error processing file: {e}") 