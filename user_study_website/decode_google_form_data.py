import pandas as pd
import json
import base64
import sys
import argparse
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

def decompress_state(compressed_data: str) -> Dict[str, Any]:
    """
    Decompress the base64-encoded state data from the Google Form
    
    Args:
        compressed_data: Base64-encoded JSON string
        
    Returns:
        Dictionary containing the decompressed data
    """
    try:
        # Decode base64
        json_data = base64.b64decode(compressed_data).decode('utf-8')
        decompressed_data = json.loads(json_data)
        
        # Extract answers and intake data
        minimal_answers = []
        intake_data = None
        
        if isinstance(decompressed_data, list):
            # Legacy format (just an array of answers)
            minimal_answers = decompressed_data
        else:
            # New format (object with answers and intake data)
            minimal_answers = decompressed_data.get('answers', [])
            intake_data = decompressed_data.get('intakeData')
        
        return {
            'answers': minimal_answers, 
            'intakeData': intake_data
        }
    except Exception as e:
        print(f"Error decompressing data: {e}")
        return {'answers': [], 'intakeData': None}

def answers_to_dataframe(decompressed_data: Dict[str, Any], participant_id: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Convert the decompressed answers data to a pandas DataFrame
    
    Args:
        decompressed_data: Dictionary containing answers and intakeData
        participant_id: Optional ID to identify this participant
        
    Returns:
        Tuple of (DataFrame with meal preferences, intake data dictionary or None)
    """
    answers = decompressed_data.get('answers', [])
    intake_data = decompressed_data.get('intakeData')
    
    if not answers:
        return pd.DataFrame(), intake_data
    
    # Transform the answers into a flat structure for the DataFrame
    rows = []
    
    for i, answer in enumerate(answers):
        row = {'meal_number': i+1}
        
        # Add participant ID if provided
        if participant_id:
            row['participant_id'] = participant_id
        
        # Extract all keys and values
        for key, value in answer.items():
            if key == 'occlusion':
                # Handle occlusion data specially
                if isinstance(value, dict):
                    row['relevant_pois'] = ','.join(value.get('r', []) if value.get('r') else [])
                    row['occluded_pois'] = ','.join(value.get('o', []) if value.get('o') else [])
            elif key == 'isOptionAPersonalized':
                # Direct value
                row['is_option_a_personalized'] = value
            elif key == 'preference_rating':
                # Extract the preference rating value
                if isinstance(value, dict) and 'value' in value:
                    row['preference_rating'] = value['value']
                else:
                    row['preference_rating'] = value
            else:
                # For regular questions, extract the value
                if isinstance(value, dict) and 'value' in value:
                    row[key] = value['value']
                else:
                    row[key] = value
        
        rows.append(row)
    
    # Create DataFrame from the rows
    df = pd.DataFrame(rows)
    
    # Add interpretations of preference ratings
    if 'preference_rating' in df.columns and 'is_option_a_personalized' in df.columns:
        def interpret_rating(row):
            rating = int(row['preference_rating']) if pd.notna(row['preference_rating']) else None
            is_a_personalized = row['is_option_a_personalized']
            
            if rating is None:
                return None
            
            # 1-3 means prefer A, 5-7 means prefer B, 4 is neutral
            if rating <= 3:
                preferred = 'Personalized' if is_a_personalized else 'Default'
            elif rating >= 5:
                preferred = 'Default' if is_a_personalized else 'Personalized'
            else:
                preferred = 'Neutral'
                
            return preferred
            
        df['preferred_option'] = df.apply(interpret_rating, axis=1)
    
    # Sort and organize columns
    important_cols = ['participant_id', 'meal_number', 'preference_rating', 'preferred_option', 'is_option_a_personalized']
    other_cols = [col for col in df.columns if col not in important_cols]
    all_cols = important_cols + sorted(other_cols)
    
    # Reorder columns (only include those that exist)
    existing_cols = [col for col in all_cols if col in df.columns]
    df = df[existing_cols]
    
    # Add participant ID to intake data if provided
    if intake_data and participant_id:
        intake_data['participant_id'] = participant_id
    
    return df, intake_data

def format_intake_data(intake_data: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """
    Format the intake data as a DataFrame with a single row
    
    Args:
        intake_data: Dictionary of intake data or None
        
    Returns:
        DataFrame with a single row containing the intake data
    """
    if not intake_data:
        return pd.DataFrame()
    
    # Create a single-row DataFrame
    return pd.DataFrame([intake_data])

def process_google_form_data(compressed_data: str, participant_id: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process the compressed data from Google Forms
    
    Args:
        compressed_data: Base64-encoded data string
        participant_id: Optional ID to identify this participant
    
    Returns:
        Tuple of (meal preferences DataFrame, intake data DataFrame)
    """
    decompressed_data = decompress_state(compressed_data)
    df_answers, intake_data = answers_to_dataframe(decompressed_data, participant_id)
    df_intake = format_intake_data(intake_data)
    
    return df_answers, df_intake

def process_multi_line_file(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process a text file where each line contains base64-encoded data for one participant
    
    Args:
        file_path: Path to the text file with multiple encoded data points
        
    Returns:
        Tuple of (combined meal preferences DataFrame, combined intake data DataFrame)
    """
    all_answers = []
    all_intake = []
    
    try:
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        print(f"Found {len(lines)} participants in file {file_path}")
        
        for i, line in enumerate(lines):
            # Use the line number as participant ID if not in the data
            participant_id = f"participant_{i+1}"
            
            try:
                df_answers, df_intake = process_google_form_data(line, participant_id)
                
                if not df_answers.empty:
                    all_answers.append(df_answers)
                    
                if not df_intake.empty:
                    # Check if participant_id already exists in the intake data
                    if 'participant_id' not in df_intake.columns:
                        df_intake['participant_id'] = participant_id
                    all_intake.append(df_intake)
                    
            except Exception as e:
                print(f"Error processing participant {i+1}: {e}")
                
        # Combine all dataframes
        if all_answers:
            combined_answers = pd.concat(all_answers, ignore_index=True)
        else:
            combined_answers = pd.DataFrame()
            
        if all_intake:
            combined_intake = pd.concat(all_intake, ignore_index=True)
        else:
            combined_intake = pd.DataFrame()
            
        return combined_answers, combined_intake
        
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return pd.DataFrame(), pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description='Decode Google Form data into pandas DataFrame')
    parser.add_argument('--data', type=str, help='Base64-encoded data from Google Form')
    parser.add_argument('--file', type=str, help='Path to file containing Base64-encoded data')
    parser.add_argument('--multi', action='store_true', help='Process file as multi-line with one encoded data point per line')
    parser.add_argument('--output', type=str, help='Output Excel file path (optional)')
    
    args = parser.parse_args()
    
    # Process based on input type
    if args.data:
        # Process single data string
        df_answers, df_intake = process_google_form_data(args.data)
    elif args.file:
        if args.multi:
            # Process multi-line file
            df_answers, df_intake = process_multi_line_file(args.file)
        else:
            # Process single-line file
            try:
                with open(args.file, 'r') as f:
                    compressed_data = f.read().strip()
                df_answers, df_intake = process_google_form_data(compressed_data)
            except Exception as e:
                print(f"Error reading file: {e}")
                return
    else:
        print("Please provide either --data or --file argument")
        return
    
    # Print summary
    print("\n=== Meal Preferences Data ===")
    if 'participant_id' in df_answers.columns:
        participant_count = df_answers['participant_id'].nunique()
        print(f"Found {participant_count} participants with {len(df_answers)} total meal responses")
    else:
        print(f"Found {len(df_answers)} meal responses")
        
    if not df_answers.empty:
        # If there are many rows, show a summary instead of the full DataFrame
        if len(df_answers) > 10:
            print("\nSample of data (first 5 rows):")
            print(df_answers.head())
            
            # Group by participant if participant_id exists
            if 'participant_id' in df_answers.columns:
                print("\nSummary by participant:")
                summary = df_answers.groupby('participant_id').agg(
                    meal_count=('meal_number', 'count'),
                    avg_preference=('preference_rating', lambda x: pd.to_numeric(x, errors='coerce').mean())
                )
                print(summary)
        else:
            print(df_answers)
    else:
        print("No meal preference data found")
    
    print("\n=== Intake Data ===")
    if not df_intake.empty:
        print(f"Found {len(df_intake)} participant intake records")
        print(df_intake)
    else:
        print("No intake data found")
    
    # Save to Excel if output file specified
    if args.output:
        try:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with pd.ExcelWriter(output_path) as writer:
                df_answers.to_excel(writer, sheet_name='Meal Preferences', index=False)
                if not df_intake.empty:
                    df_intake.to_excel(writer, sheet_name='Participant Info', index=False)
            print(f"\nData saved to: {output_path}")
        except Exception as e:
            print(f"Error saving Excel file: {e}")

if __name__ == "__main__":
    main() 