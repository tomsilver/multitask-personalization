#!/usr/bin/env python3
"""
Example script demonstrating how to use the meal preferences decoder and analyzer.
This creates a sample dataset, encodes it, then decodes and analyzes it to demonstrate
the full workflow.
"""

import json
import base64
import pandas as pd
from pathlib import Path
from decode_google_form_data import process_google_form_data
from analyze_preferences import analyze_preferences, analyze_intake_data

# Create a sample dataset to demonstrate the workflow
def create_sample_data():
    """Create sample data that mimics the format from the Google Form"""
    
    # Sample answers for 5 meals
    sample_answers = []
    
    for i in range(5):
        meal_num = i + 1
        
        # Basic answer structure
        answer = {
            # Whether option A is personalized (alternating)
            "isOptionAPersonalized": i % 2 == 0,
            
            # Preference rating (varying from 1-7)
            "preference_rating": {
                "value": str(((i + 2) % 7) + 1),
                "metadata": None
            },
            
            # Meal-specific preferences
            "bite_order": {
                "value": "clockwise" if i % 3 == 0 else "counterclockwise" if i % 3 == 1 else "random",
                "metadata": None
            },
            
            "ready_signal": {
                "value": "head_nod" if i % 2 == 0 else "verbal_cue",
                "metadata": None
            },
            
            "verbal": {
                "value": "True" if i % 2 == 0 else "False",
                "metadata": None
            },
            
            # Occlusion data
            "look_forward": {
                "value": "Yes",
                "metadata": None
            },
            
            "block_forward": {
                "value": "No" if i < 3 else "Yes",
                "metadata": None
            },
            
            "look_left": {
                "value": "Yes" if i % 2 == 1 else "No",
                "metadata": None
            },
            
            "block_left": {
                "value": "No",
                "metadata": None
            },
            
            # Occlusion structure
            "occlusion": {
                "relevant_pois": ["front"] if i % 2 == 0 else ["front", "left"],
                "occluded_pois": [] if i < 3 else ["front"]
            }
        }
        
        sample_answers.append(answer)
    
    # Sample intake data
    sample_intake = {
        "name": "Test Participant",
        "age": "30",
        "gender": "Other",
        "robotExperience": "Some",
        "fedExperience": "None"
    }
    
    # Create compressed data using the same format as the app.js
    compressed_data = {
        "answers": sample_answers,
        "intakeData": sample_intake
    }
    
    # Convert to JSON and encode with base64
    json_data = json.dumps(compressed_data)
    encoded_data = base64.b64encode(json_data.encode()).decode()
    
    return encoded_data

def run_example():
    """Run the full example workflow"""
    
    print("=" * 80)
    print("MEAL PREFERENCES ANALYSIS EXAMPLE")
    print("=" * 80)
    
    # Step 1: Create sample data
    print("\nStep 1: Creating sample data...")
    encoded_data = create_sample_data()
    print(f"Generated base64-encoded data ({len(encoded_data)} chars)")
    
    # Step 2: Decode the data
    print("\nStep 2: Decoding the data...")
    df_answers, df_intake = process_google_form_data(encoded_data)
    
    print("\n--- Meal Preferences Data ---")
    print(df_answers)
    
    print("\n--- Participant Data ---")
    print(df_intake)
    
    # Step 3: Save to Excel
    output_dir = Path("example_output")
    output_dir.mkdir(exist_ok=True)
    
    excel_path = output_dir / "example_data.xlsx"
    print(f"\nStep 3: Saving data to Excel: {excel_path}")
    
    with pd.ExcelWriter(excel_path) as writer:
        df_answers.to_excel(writer, sheet_name='Meal Preferences', index=False)
        df_intake.to_excel(writer, sheet_name='Participant Info', index=False)
    
    # Step 4: Analyze the data
    print("\nStep 4: Analyzing the data...")
    analyze_preferences(df_answers, output_dir)
    analyze_intake_data(df_intake)
    
    print("\nExample completed successfully!")
    print(f"Check {output_dir} for the output files.")

if __name__ == "__main__":
    run_example() 