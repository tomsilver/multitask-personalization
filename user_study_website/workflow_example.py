#!/usr/bin/env python3
"""
Complete workflow example demonstrating how to use all the meal preferences analysis tools together.

This script:
1. Creates sample data for multiple participants
2. Saves each participant's data to individual files
3. Creates a combined file with all participants (one per line)
4. Processes individual files and the combined file
5. Generates individual and combined analyses

Usage:
    python workflow_example.py
"""

import os
import base64
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

# Import functionality from our other scripts
from example import create_sample_data
from decode_google_form_data import process_google_form_data, process_multi_line_file
from analyze_preferences import analyze_preferences, analyze_intake_data
from batch_process import process_file, analyze_combined_data

# Define the number of participants to simulate
NUM_PARTICIPANTS = 5

def create_multi_participant_sample():
    """Create sample data for multiple participants with variations"""
    
    participant_data = []
    
    for i in range(NUM_PARTICIPANTS):
        # Create base data
        encoded_data = create_sample_data()
        
        # Add some variations based on participant number
        # (In real data, there would be more significant differences)
        participant_id = f"participant_{i+1}"
        
        # Decode the data
        json_data = base64.b64decode(encoded_data).decode('utf-8')
        data_dict = json.loads(json_data)
        
        # Modify the participant-specific information
        if 'intakeData' in data_dict:
            data_dict['intakeData']['name'] = f"Test Participant {i+1}"
            data_dict['intakeData']['age'] = str(25 + i * 5)  # Vary ages
            
            # Vary gender
            genders = ["Male", "Female", "Other", "Female", "Male"]
            data_dict['intakeData']['gender'] = genders[i]
            
            # Vary robot experience
            experiences = ["None", "Some", "Extensive", "Some", "None"]
            data_dict['intakeData']['robotExperience'] = experiences[i]
        
        # Vary meal preferences
        if 'answers' in data_dict and len(data_dict['answers']) > 0:
            for j, answer in enumerate(data_dict['answers']):
                # Vary preference ratings based on participant
                if 'preference_rating' in answer:
                    # Create some variation patterns
                    if i % 2 == 0:  # Even-numbered participants prefer personalized
                        preference = max(1, min(7, 7 - j - i % 3))  # Decreasing preference (1-7)
                    else:  # Odd-numbered participants prefer default
                        preference = max(1, min(7, 1 + j + i % 3))  # Increasing preference (1-7)
                    
                    if isinstance(answer['preference_rating'], dict):
                        answer['preference_rating']['value'] = str(preference)
                    else:
                        answer['preference_rating'] = str(preference)
        
        # Re-encode the modified data
        json_data = json.dumps(data_dict)
        encoded_data = base64.b64encode(json_data.encode()).decode()
        
        participant_data.append({
            'id': participant_id,
            'encoded_data': encoded_data
        })
    
    return participant_data

def run_workflow():
    """Run the complete workflow"""
    
    print("=" * 80)
    print("COMPLETE MEAL PREFERENCES ANALYSIS WORKFLOW EXAMPLE")
    print("=" * 80)
    
    # Create output directories
    output_base = Path("workflow_example_output")
    if output_base.exists():
        shutil.rmtree(output_base)
    
    individual_data_dir = output_base / "individual_data"
    combined_data_dir = output_base / "combined_data"
    analysis_dir = output_base / "analysis"
    excel_dir = output_base / "excel_output"
    
    for directory in [output_base, individual_data_dir, combined_data_dir, analysis_dir, excel_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create sample data for multiple participants
    print("\nStep 1: Creating sample data for multiple participants...")
    participant_data = create_multi_participant_sample()
    print(f"Generated data for {len(participant_data)} participants")
    
    # Step 2: Save individual files
    print("\nStep 2: Saving individual participant data files...")
    for participant in participant_data:
        file_path = individual_data_dir / f"{participant['id']}.txt"
        with open(file_path, 'w') as f:
            f.write(participant['encoded_data'])
        print(f"Saved data for {participant['id']} to {file_path}")
    
    # Step 3: Create combined file
    print("\nStep 3: Creating combined multi-participant file...")
    combined_file_path = combined_data_dir / "all_participants.txt"
    with open(combined_file_path, 'w') as f:
        for participant in participant_data:
            f.write(participant['encoded_data'] + '\n')
    print(f"Saved combined data to {combined_file_path}")
    
    # Step 4: Process individual files
    print("\nStep 4: Processing individual participant files...")
    for participant in participant_data:
        file_path = individual_data_dir / f"{participant['id']}.txt"
        output_path = excel_dir / f"{participant['id']}.xlsx"
        
        # Decode data
        df_answers, df_intake = process_file(file_path)
        
        # Save to Excel
        with pd.ExcelWriter(output_path) as writer:
            df_answers.to_excel(writer, sheet_name='Meal Preferences', index=False)
            if not df_intake.empty:
                df_intake.to_excel(writer, sheet_name='Participant Info', index=False)
        
        print(f"Processed and saved {participant['id']} data to {output_path}")
    
    # Step 5: Process combined file with multi-line option
    print("\nStep 5: Processing combined multi-participant file...")
    multi_output_path = excel_dir / "all_participants.xlsx"
    
    # Decode data with multi-line processing
    df_answers, df_intake = process_multi_line_file(combined_file_path)
    
    # Save to Excel
    with pd.ExcelWriter(multi_output_path) as writer:
        df_answers.to_excel(writer, sheet_name='Meal Preferences', index=False)
        if not df_intake.empty:
            df_intake.to_excel(writer, sheet_name='Participant Info', index=False)
    
    print(f"Processed and saved combined data to {multi_output_path}")
    
    # Step 6: Generate individual analyses
    print("\nStep 6: Generating individual participant analyses...")
    for participant in participant_data:
        participant_id = participant['id']
        file_path = individual_data_dir / f"{participant_id}.txt"
        output_dir = analysis_dir / participant_id
        
        # Decode data
        with open(file_path, 'r') as f:
            encoded_data = f.read().strip()
        
        df_answers, df_intake = process_google_form_data(encoded_data, participant_id)
        
        # Generate analysis
        analyze_preferences(df_answers, output_dir)
        
        print(f"Generated analysis for {participant_id} in {output_dir}")
    
    # Step 7: Generate combined analysis
    print("\nStep 7: Generating combined analysis...")
    combined_analysis_dir = analysis_dir / "combined"
    analyze_combined_data(df_answers, df_intake, combined_analysis_dir)
    print(f"Generated combined analysis in {combined_analysis_dir}")
    
    print("\nWorkflow completed successfully!")
    print(f"All outputs can be found in {output_base}")
    print("\nExample commands to run these steps manually:")
    print(f"  1. python decode_google_form_data.py --file {combined_file_path} --multi --output {multi_output_path}")
    print(f"  2. python batch_process.py --input_dir {individual_data_dir} --output_dir {analysis_dir}")
    print(f"  3. python batch_process.py --input_dir {excel_dir} --output_dir {analysis_dir}/excel_based --process_excel")

if __name__ == "__main__":
    run_workflow() 