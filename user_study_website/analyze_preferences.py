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

def process_preferences(responses):
    """
    Process responses to extract preference ratings and build a dataframe.
    
    Args:
        responses: List of response objects
        
    Returns:
        pandas DataFrame with processed preference data
    """
    processed_data = []
    
    for response_idx, response in enumerate(responses):
        participant_id = response_idx + 1
        participant_info = response.get('participantInfo', {})
        participant_name = participant_info.get('name', f"Participant_{participant_id}")
        
        answers = response.get('answers', [])
        option_mappings_str = response.get('optionMappings', "")
        
        # Parse the option mappings (format: "1:A,2:B,3:A,4:A,5:B")
        option_mappings = {}
        if option_mappings_str:
            for mapping in option_mappings_str.split(','):
                parts = mapping.split(':')
                if len(parts) == 2:
                    meal_num = int(parts[0])
                    is_option_a_personalized = parts[1] == 'A'
                    option_mappings[meal_num] = is_option_a_personalized
        
        # Process each meal's answers
        for meal_idx, meal_answer in enumerate(answers):
            meal_num = meal_idx + 1
            
            # Skip if missing essential data
            if 'preference_rating' not in meal_answer:
                continue
                
            # Get raw preference rating (1-7 scale)
            raw_rating = int(meal_answer['preference_rating']['value'])
            
            # Determine if personalized option was A or B based on option mappings
            is_option_a_personalized = option_mappings.get(meal_num, 
                                                          meal_answer.get('isOptionAPersonalized', False))
            
            # Adjust rating to reflect preference for personalized option
            # If option A is personalized:
            #   - Ratings 1-3 (prefer A) become 7-5 (prefer personalized)
            #   - Rating 4 (neutral) stays 4
            #   - Ratings 5-7 (prefer B) become 3-1 (dislike personalized)
            # If option B is personalized:
            #   - Ratings 1-3 (prefer A) become 1-3 (dislike personalized)
            #   - Rating 4 (neutral) stays 4
            #   - Ratings 5-7 (prefer B) become 5-7 (prefer personalized)
            
            if is_option_a_personalized:
                if raw_rating < 4:  # Prefer A (personalized)
                    preference_for_personalized = 8 - raw_rating  # 1->7, 2->6, 3->5
                elif raw_rating == 4:  # Neutral
                    preference_for_personalized = 4
                else:  # Prefer B (non-personalized)
                    preference_for_personalized = 8 - raw_rating  # 5->3, 6->2, 7->1
            else:  # Option B is personalized
                preference_for_personalized = raw_rating  # Already aligned correctly
            
            # Create a row for this meal's data
            row = {
                'participant_id': participant_id,
                'participant_name': participant_name,
                'meal_number': meal_num,
                'raw_preference_rating': raw_rating,
                'personalized_option': 'A' if is_option_a_personalized else 'B',
                'preference_for_personalized': preference_for_personalized
            }
            
            # Add demographic info if available
            if participant_info:
                row.update({
                    'age': participant_info.get('age', None),
                    'gender': participant_info.get('gender', None),
                    'robot_experience': participant_info.get('robotExp', None),
                    'fed_experience': participant_info.get('fedExp', None)
                })
            
            processed_data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(processed_data)
    return df

def plot_preferences(df):
    """
    Create plots showing preference for personalized options across meals.
    
    Args:
        df: pandas DataFrame with processed preference data
    """
    # Set the style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Create the main plot - preference for personalized by meal number
    ax = sns.boxplot(
        x='meal_number', 
        y='preference_for_personalized', 
        data=df,
        palette='Blues'
    )
    
    # Add individual data points with jitter
    sns.stripplot(
        x='meal_number', 
        y='preference_for_personalized', 
        data=df,
        color='navy', 
        alpha=0.5, 
        jitter=True
    )
    
    # Add a horizontal line at the neutral point (4)
    plt.axhline(y=4, color='gray', linestyle='--', alpha=0.7)
    
    # Customize the plot
    plt.title('Preference for Personalized Option by Meal Number', fontsize=16)
    plt.xlabel('Meal Number', fontsize=14)
    plt.ylabel('Preference for Personalized Option (1-7)', fontsize=14)
    plt.ylim(0.5, 7.5)  # Set y-axis limits
    
    # Add text labels to explain the scale
    plt.text(df['meal_number'].max() + 0.3, 1, 'Strongly Dislike', 
             va='center', ha='left', fontsize=10, color='gray')
    plt.text(df['meal_number'].max() + 0.3, 4, 'Neutral', 
             va='center', ha='left', fontsize=10, color='gray')
    plt.text(df['meal_number'].max() + 0.3, 7, 'Strongly Prefer', 
             va='center', ha='left', fontsize=10, color='gray')
    
    # Calculate mean preference for each meal
    mean_prefs = df.groupby('meal_number')['preference_for_personalized'].mean()
    
    # Add mean values as text annotations
    for i, mean_val in enumerate(mean_prefs):
        plt.text(i, mean_val + 0.2, f'Mean: {mean_val:.2f}', ha='center', color='navy', fontweight='bold')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('personalized_preference_by_meal.png', dpi=300)
    
    # Create a second plot - average preference trend
    plt.figure(figsize=(10, 6))
    
    # Line plot with error bars showing the trend over meals
    sns.lineplot(
        x='meal_number', 
        y='preference_for_personalized', 
        data=df,
        marker='o',
        err_style='band',
        ci=95,
        color='navy'
    )
    
    # Add a horizontal line at the neutral point (4)
    plt.axhline(y=4, color='gray', linestyle='--', alpha=0.7)
    
    # Customize the plot
    plt.title('Trend of Preference for Personalized Option', fontsize=16)
    plt.xlabel('Meal Number', fontsize=14)
    plt.ylabel('Average Preference for Personalized Option (1-7)', fontsize=14)
    plt.ylim(0.5, 7.5)  # Set y-axis limits
    plt.xticks(df['meal_number'].unique())
    
    # Save the trend plot
    plt.tight_layout()
    plt.savefig('personalized_preference_trend.png', dpi=300)
    
    print("Plots created: personalized_preference_by_meal.png and personalized_preference_trend.png")

def generate_summary_statistics(df):
    """
    Generate summary statistics for the preference data
    
    Args:
        df: pandas DataFrame with processed preference data
        
    Returns:
        pandas DataFrame with summary statistics
    """
    # Overall statistics
    overall_mean = df['preference_for_personalized'].mean()
    overall_median = df['preference_for_personalized'].median()
    
    print(f"Overall preference for personalized options: Mean = {overall_mean:.2f}, Median = {overall_median:.2f}")
    
    # By meal statistics
    meal_stats = df.groupby('meal_number')['preference_for_personalized'].agg(['mean', 'median', 'std', 'count'])
    meal_stats.columns = ['Mean', 'Median', 'Std Dev', 'Count']
    
    print("\nPreference for personalized option by meal:")
    print(meal_stats)
    
    # Save statistics to CSV
    meal_stats.to_csv('meal_preference_statistics.csv')
    print("Statistics saved to meal_preference_statistics.csv")
    
    return meal_stats

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Analyze meal preference data from Google Form responses.')
    parser.add_argument('file_path', nargs='?', default='form_responses.txt',
                        help='Path to the form responses file (default: form_responses.txt)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print verbose information during processing')
    
    # Parse arguments
    args = parser.parse_args()
    file_path = Path(args.file_path)
    verbose = args.verbose
    
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist.")
        return
    
    try:
        # Load and process the data
        print(f"Loading responses from {file_path}...")
        responses = load_responses(file_path)
        print(f"Loaded {len(responses)} responses.")
        
        if not responses:
            print("No valid responses found. Exiting.")
            return
            
        print("Processing preference data...")
        df = process_preferences(responses)
        
        if verbose:
            print("\nFirst few rows of processed data:")
            print(df.head())
        
        # Generate statistics
        generate_summary_statistics(df)
        
        # Create plots
        print("Creating plots...")
        plot_preferences(df)
        
        print("Analysis complete!")
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 