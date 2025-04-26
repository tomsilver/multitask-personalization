#!/usr/bin/env python3
"""
Batch processor for meal preferences data.
This script processes multiple base64-encoded data files and creates both
individual and combined analyses.
"""

import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from decode_google_form_data import process_google_form_data, process_multi_line_file


def process_file(file_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process a single data file
    
    Args:
        file_path: Path to the file containing base64-encoded data
        
    Returns:
        Tuple of (meal preferences DataFrame, intake data DataFrame)
    """
    print(f"Processing file: {file_path}")
    
    try:
        # Handle different file types
        if file_path.suffix.lower() == '.xlsx' or file_path.suffix.lower() == '.xls':
            # Excel file - likely already processed by decode_google_form_data.py
            return process_excel_file(file_path)
        else:
            # Assume text file with base64 data
            with open(file_path, 'r') as f:
                compressed_data = f.read().strip()
            
            # Add participant ID based on filename
            participant_id = file_path.stem
            
            # Process the data
            df_answers, df_intake = process_google_form_data(compressed_data, participant_id)
            
            return df_answers, df_intake
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return pd.DataFrame(), pd.DataFrame()


def process_excel_file(file_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process an Excel file that contains already decoded data
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        Tuple of (meal preferences DataFrame, intake data DataFrame)
    """
    try:
        # Load meal preferences sheet
        df_answers = pd.read_excel(file_path, sheet_name='Meal Preferences')
        
        # Try to load intake data if available
        try:
            df_intake = pd.read_excel(file_path, sheet_name='Participant Info')
        except:
            print(f"No participant info sheet found in {file_path}")
            df_intake = pd.DataFrame()
        
        # Ensure participant_id exists
        if 'participant_id' not in df_answers.columns:
            # Use filename as participant_id
            participant_id = file_path.stem
            df_answers['participant_id'] = participant_id
            
            if not df_intake.empty and 'participant_id' not in df_intake.columns:
                df_intake['participant_id'] = participant_id
        
        print(f"Loaded Excel file with {len(df_answers)} meal entries and {len(df_intake)} participant records")
        return df_answers, df_intake
        
    except Exception as e:
        print(f"Error loading Excel file {file_path}: {e}")
        return pd.DataFrame(), pd.DataFrame()


def combine_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine multiple DataFrames into one
    
    Args:
        dfs: List of DataFrames to combine
        
    Returns:
        Combined DataFrame
    """
    # Filter out empty DataFrames
    valid_dfs = [df for df in dfs if not df.empty]
    
    if not valid_dfs:
        return pd.DataFrame()
    
    return pd.concat(valid_dfs, ignore_index=True)


def analyze_combined_data(combined_answers: pd.DataFrame, combined_intake: pd.DataFrame, output_dir: Path):
    """
    Analyze the combined dataset from multiple participants
    
    Args:
        combined_answers: Combined meal preferences DataFrame
        combined_intake: Combined intake data DataFrame
        output_dir: Directory to save the visualizations
    """
    if combined_answers.empty:
        print("No valid data to analyze")
        return
    
    # Set up plotting style
    setup_plotting_style()
    
    # Create output directory if it doesn't exist
    output_path = create_output_dir(output_dir)
    
    print(f"\nGenerating combined analysis for {combined_answers['participant_id'].nunique()} participants...")
    
    # 1. Distribution of preference ratings across all participants
    plt.figure(figsize=(10, 6))
    sns.countplot(x='preference_rating', data=combined_answers, 
                  order=sorted(combined_answers['preference_rating'].unique()))
    plt.title('Distribution of Preference Ratings Across All Participants')
    plt.xlabel('Preference Rating (1=Strongly prefer A, 7=Strongly prefer B)')
    plt.ylabel('Count')
    plt.savefig(output_path / 'combined_preference_ratings.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Preference for personalized vs default across all participants
    if 'preferred_option' in combined_answers.columns:
        plt.figure(figsize=(8, 6))
        preference_counts = combined_answers['preferred_option'].value_counts()
        plt.pie(preference_counts, labels=preference_counts.index, autopct='%1.1f%%', 
                startangle=90, shadow=False)
        plt.axis('equal')
        plt.title('Overall Preference for Personalized vs Default Options')
        plt.savefig(output_path / 'combined_personalized_preference.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Preference evolution across meals (averaged across participants)
    if 'meal_number' in combined_answers.columns and 'preferred_option' in combined_answers.columns:
        plt.figure(figsize=(12, 6))
        
        # Get preference counts per meal
        pref_by_meal = combined_answers.groupby(['meal_number', 'preferred_option']).size().unstack(fill_value=0)
        
        # Convert to percentages
        pref_by_meal_pct = pref_by_meal.div(pref_by_meal.sum(axis=1), axis=0) * 100
        
        # Plot as stacked bar chart
        pref_by_meal_pct.plot(kind='bar', stacked=True)
        plt.title('Evolution of Preferences Across Meals (All Participants)')
        plt.xlabel('Meal Number')
        plt.ylabel('Percentage of Participants')
        plt.xticks(rotation=0)
        plt.legend(title='Preference')
        plt.savefig(output_path / 'combined_preference_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Participant comparison (preference for personalized by participant)
    if 'participant_id' in combined_answers.columns and 'preferred_option' in combined_answers.columns:
        plt.figure(figsize=(12, 6))
        
        # Calculate percentage of personalized preference by participant
        participant_prefs = combined_answers.groupby('participant_id')['preferred_option'].apply(
            lambda x: (x == 'Personalized').mean() * 100
        ).sort_values(ascending=False)
        
        # Plot as horizontal bar chart
        participant_prefs.plot(kind='barh')
        plt.title('Preference for Personalized Options by Participant')
        plt.xlabel('Percentage of Meals Where Personalized Option Was Preferred')
        plt.ylabel('Participant ID')
        plt.axvline(x=50, color='red', linestyle='--', alpha=0.7)
        plt.grid(axis='x', alpha=0.3)
        plt.savefig(output_path / 'participant_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Summary table of participant demographics
    if not combined_intake.empty:
        # Create a summary of demographics
        demo_summary = {
            'Total Participants': len(combined_intake),
            'Average Age': pd.to_numeric(combined_intake['age'], errors='coerce').mean(),
            'Gender Distribution': combined_intake['gender'].value_counts().to_dict(),
            'Robot Experience': combined_intake['robotExperience'].value_counts().to_dict(),
            'Fed Experience': combined_intake['fedExperience'].value_counts().to_dict()
        }
        
        # Save demographic summary to a text file
        with open(output_path / 'demographic_summary.txt', 'w') as f:
            f.write("PARTICIPANT DEMOGRAPHIC SUMMARY\n")
            f.write("===============================\n\n")
            f.write(f"Total Participants: {demo_summary['Total Participants']}\n")
            f.write(f"Average Age: {demo_summary['Average Age']:.1f}\n\n")
            
            f.write("Gender Distribution:\n")
            for gender, count in demo_summary['Gender Distribution'].items():
                f.write(f"  - {gender}: {count} ({count/demo_summary['Total Participants']*100:.1f}%)\n")
            
            f.write("\nRobot Experience:\n")
            for exp, count in demo_summary['Robot Experience'].items():
                f.write(f"  - {exp}: {count} ({count/demo_summary['Total Participants']*100:.1f}%)\n")
            
            f.write("\nFeeding Experience:\n")
            for exp, count in demo_summary['Fed Experience'].items():
                f.write(f"  - {exp}: {count} ({count/demo_summary['Total Participants']*100:.1f}%)\n")
    
    # 6. Save combined data to Excel
    with pd.ExcelWriter(output_path / 'combined_data.xlsx') as writer:
        combined_answers.to_excel(writer, sheet_name='All Meal Preferences', index=False)
        if not combined_intake.empty:
            combined_intake.to_excel(writer, sheet_name='All Participants', index=False)
    
    print(f"Combined analysis saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Batch process multiple meal preference data files')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, default='batch_results', help='Directory to save the results')
    parser.add_argument('--individual', action='store_true', help='Generate individual analyses for each participant')
    parser.add_argument('--combined', action='store_true', help='Generate combined analysis across all participants')
    parser.add_argument('--file_pattern', type=str, default='*.txt', help='File pattern to match (default: *.txt)')
    parser.add_argument('--process_excel', action='store_true', help='Process Excel files (*.xlsx, *.xls) in addition to text files')
    
    args = parser.parse_args()
    
    # Default to both if neither is specified
    if not args.individual and not args.combined:
        args.individual = True
        args.combined = True
    
    # Create base output directory
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all matching files in the input directory
    input_dir = Path(args.input_dir)
    
    # Include Excel files if specified
    patterns = [args.file_pattern]
    if args.process_excel or args.file_pattern.endswith('.xlsx') or args.file_pattern.endswith('.xls'):
        if args.file_pattern == '*.txt':  # If default pattern, also look for Excel files
            patterns.extend(['*.xlsx', '*.xls'])
    
    # Find all matching files
    data_files = []
    for pattern in patterns:
        data_files.extend(list(input_dir.glob(pattern)))
    
    if not data_files:
        print(f"No matching files found in {input_dir} with patterns {patterns}")
        return
    
    print(f"Found {len(data_files)} data files to process")
    
    # Process each file
    all_answers_dfs = []
    all_intake_dfs = []
    
    for file_path in data_files:
        # Process the file
        df_answers, df_intake = process_file(file_path)
        
        # Check if this is a multi-participant dataset
        if not df_answers.empty and 'participant_id' in df_answers.columns:
            participant_ids = df_answers['participant_id'].unique()
            print(f"Found {len(participant_ids)} participants in file {file_path}")
            
            # Individual analysis for each participant if requested
            if args.individual:
                for participant_id in participant_ids:
                    # Filter data for this participant
                    participant_answers = df_answers[df_answers['participant_id'] == participant_id].copy()
                    
                    # Filter intake data if available
                    if not df_intake.empty and 'participant_id' in df_intake.columns:
                        participant_intake = df_intake[df_intake['participant_id'] == participant_id].copy()
                    else:
                        participant_intake = pd.DataFrame()
                    
                    # Create output directory for this participant
                    individual_output_dir = base_output_dir / str(participant_id)
                    
                    print(f"Generating individual analysis for participant {participant_id}...")
                    from analyze_preferences import analyze_preferences, analyze_intake_data
                    analyze_preferences(participant_answers, individual_output_dir)
                    if not participant_intake.empty:
                        analyze_intake_data(participant_intake)
        
        # Individual analysis if requested (for single-participant files)
        elif args.individual and not df_answers.empty:
            participant_id = file_path.stem if 'participant_id' not in df_answers.columns else df_answers['participant_id'].iloc[0]
            individual_output_dir = base_output_dir / participant_id
            print(f"Generating individual analysis for participant {participant_id}...")
            from analyze_preferences import analyze_preferences, analyze_intake_data
            analyze_preferences(df_answers, individual_output_dir)
            if not df_intake.empty:
                analyze_intake_data(df_intake)
        
        # Add to lists for combined analysis
        all_answers_dfs.append(df_answers)
        all_intake_dfs.append(df_intake)
    
    # Combined analysis if requested
    if args.combined:
        combined_answers = combine_dataframes(all_answers_dfs)
        combined_intake = combine_dataframes(all_intake_dfs)
        
        combined_output_dir = base_output_dir / 'combined'
        analyze_combined_data(combined_answers, combined_intake, combined_output_dir)
    
    print("\nBatch processing completed successfully!")


if __name__ == "__main__":
    # Import these at the bottom to avoid circular imports
    from analyze_preferences import setup_plotting_style, create_output_dir
    main() 