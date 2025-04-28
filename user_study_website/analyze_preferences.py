#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from scipy import stats

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
                
            # Get raw preference rating (1-5 scale)
            raw_rating = int(meal_answer['preference_rating']['value'])
            
            # Determine if personalized option was A or B based on option mappings
            is_option_a_personalized = option_mappings.get(meal_num, 
                                                          meal_answer.get('isOptionAPersonalized', False))
            
            # Adjust rating to reflect preference for personalized option
            # If option A is personalized:
            #   - Ratings 1-2 (prefer A) become 5-4 (prefer personalized)
            #   - Rating 3 (neutral) stays 3
            #   - Ratings 4-5 (prefer B) become 2-1 (dislike personalized)
            # If option B is personalized:
            #   - Ratings 1-2 (prefer A) become 1-2 (dislike personalized)
            #   - Rating 3 (neutral) stays 3
            #   - Ratings 4-5 (prefer B) become 4-5 (prefer personalized)
            
            if is_option_a_personalized:
                if raw_rating < 3:  # Prefer A (personalized)
                    preference_for_personalized = 6 - raw_rating  # 1->5, 2->4
                elif raw_rating == 3:  # Neutral
                    preference_for_personalized = 3
                else:  # Prefer B (non-personalized)
                    preference_for_personalized = 6 - raw_rating  # 4->2, 5->1
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
    # Set the style and font configurations
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook', 'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif'],
        'font.sans-serif': ['Helvetica', 'Avant Garde', 'Computer Modern Sans Serif'],
        'font.cursive': 'Zapf Chancery',
        'font.monospace': ['Courier', 'Computer Modern Typewriter'],
        'font.size': 18.0,
        'axes.titlesize': 28.0,
        'axes.titlepad': 20,
        'axes.labelsize': 'large',
        'axes.labelweight': 300,
        'lines.linewidth': 2,
        'mathtext.rm': 'serif',
        'mathtext.it': 'serif:italic',
        'mathtext.bf': 'serif:bold'
    })
    
    # Verify font settings
    print("Current font settings:")
    print(f"Font family: {plt.rcParams['font.family']}")
    print(f"Serif fonts: {plt.rcParams['font.serif']}")
    print(f"Font size: {plt.rcParams['font.size']}")
    
    # Set the style
    sns.set(style="whitegrid", font_scale=1.2)
    
    # Create the horizontal bar plot
    plt.figure(figsize=(12, 8))
    
    # Calculate mean preferences for each meal
    meal_means = df.groupby('meal_number')['preference_for_personalized'].mean().reset_index()
    meal_means['meal_label'] = meal_means['meal_number'].apply(lambda x: f'Meal {x}')
    
    # Calculate standard error for each meal
    meal_counts = df.groupby('meal_number').size()
    meal_stds = df.groupby('meal_number')['preference_for_personalized'].std()
    meal_stderr = meal_stds / np.sqrt(meal_counts)
    
    # Perform Wilcoxon signed-rank test for each meal
    significant_meals = []
    for meal_num in df['meal_number'].unique():
        meal_data = df[df['meal_number'] == meal_num]['preference_for_personalized']
        # Test against neutral value of 3
        statistic, p_value = stats.wilcoxon(meal_data - 3)
        if p_value < 0.005:
            significant_meals.append(meal_num)
    
    # Create the horizontal bar plot
    ax = sns.barplot(
        y='meal_label',
        x='preference_for_personalized',
        data=meal_means,
        palette='Blues_d',
        alpha=0.8
    )
    
    # Add a vertical line at the neutral point (3)
    plt.axvline(x=3, color='gray', linestyle='--', alpha=0.7)
    
    # Customize the plot with larger fonts
    plt.xlabel('Preference for CBTL (Ours)', fontsize=24, fontfamily='serif')
    plt.ylabel('')  # Remove y-axis label
    plt.xlim(0.5, 5.5)  # Set x-axis limits
    
    # Set custom x-axis ticks with labels
    plt.xticks([1, 2, 3, 4, 5], 
               ['Strongly\nDislike', 'Dislike', 'Neutral', 'Prefer', 'Strongly\nPrefer'],
               fontsize=18, fontfamily='serif')
    
    # Increase y-tick label size
    plt.yticks(fontsize=18, fontfamily='serif')
    
    # Add error bars using standard error
    for i, (meal_num, stderr) in enumerate(meal_stderr.items()):
        mean = meal_means.loc[i, 'preference_for_personalized']
        plt.errorbar(mean, i, xerr=stderr, color='black', capsize=5, alpha=0.5)
        
        # Add star for statistically significant meals
        if meal_num in significant_meals:
            # Position star to the right of the bar, vertically centered
            plt.text(mean + 0.3, i + 0.1, '*', 
                    fontsize=32,  # Large font size
                    color='black', 
                    ha='center',  # Center horizontally
                    va='center',  # Center vertically
                    fontweight='bold')  # Make star bolder
    
    # Save the horizontal bar plot
    plt.tight_layout()
    plt.savefig('personalized_preference_horizontal.png', dpi=300, bbox_inches='tight')
    
    # Create the original box plot
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
    
    # Add a horizontal line at the neutral point (3)
    plt.axhline(y=3, color='gray', linestyle='--', alpha=0.7)
    
    # Customize the plot
    plt.title('Preference for Personalized Option by Meal Number', fontsize=16)
    plt.xlabel('Meal Number', fontsize=14)
    plt.ylabel('Preference for Personalized Option (1-5)', fontsize=14)
    plt.ylim(0.5, 5.5)  # Set y-axis limits
    
    # Add text labels to explain the scale
    plt.text(df['meal_number'].max() + 0.3, 1, 'Strongly Dislike', 
             va='center', ha='left', fontsize=10, color='gray')
    plt.text(df['meal_number'].max() + 0.3, 3, 'Neutral', 
             va='center', ha='left', fontsize=10, color='gray')
    plt.text(df['meal_number'].max() + 0.3, 5, 'Strongly Prefer', 
             va='center', ha='left', fontsize=10, color='gray')
    
    # Calculate mean preference for each meal
    mean_prefs = df.groupby('meal_number')['preference_for_personalized'].mean()
    
    # Add mean values as text annotations
    for i, mean_val in enumerate(mean_prefs):
        plt.text(i, mean_val + 0.2, f'Mean: {mean_val:.2f}', ha='center', color='navy', fontweight='bold')
    
    # Save the box plot
    plt.tight_layout()
    plt.savefig('personalized_preference_by_meal.png', dpi=300)
    
    # Create the trend plot
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
    
    # Add a horizontal line at the neutral point (3)
    plt.axhline(y=3, color='gray', linestyle='--', alpha=0.7)
    
    # Customize the plot
    plt.title('Trend of Preference for Personalized Option', fontsize=16)
    plt.xlabel('Meal Number', fontsize=14)
    plt.ylabel('Average Preference for Personalized Option (1-5)', fontsize=14)
    plt.ylim(0.5, 5.5)  # Set y-axis limits
    plt.xticks(df['meal_number'].unique())
    
    # Save the trend plot
    plt.tight_layout()
    plt.savefig('personalized_preference_trend.png', dpi=300)
    
    print("Plots created: personalized_preference_horizontal.png, personalized_preference_by_meal.png, and personalized_preference_trend.png")

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