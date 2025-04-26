import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
from matplotlib.ticker import MaxNLocator

def load_data(csv_path):
    """
    Load the data from the CSV file created by decode_responses.py
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing the decoded responses
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded data with {len(df)} responses and {len(df.columns)} columns.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def convert_preference_to_numeric(df):
    """
    Convert preference ratings to numeric values and add derived columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing the survey responses
        
    Returns:
        pd.DataFrame: DataFrame with converted preference columns
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Check for preference columns
    pref_cols = [col for col in df.columns if 'preference' in col]
    
    for col in pref_cols:
        # Convert to numeric values
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Create a derived column indicating if the person preferred 
        # the personalized option (rating < 4) or non-personalized option (rating > 4)
        # or was neutral (rating = 4)
        meal_num = col.split('_')[0]  # Extract meal number (e.g., 'meal1')
        personalized_col = f"{meal_num}_optionA_personalized"
        
        if personalized_col in df.columns:
            # Create a new column indicating preference for personalized option
            preferred_personalized = []
            
            for i, row in df_clean.iterrows():
                rating = row[col]
                option_a_personalized = row[personalized_col]
                
                if pd.isna(rating) or pd.isna(option_a_personalized):
                    preferred_personalized.append(np.nan)
                elif rating == 4:  # Neutral
                    preferred_personalized.append('Neutral')
                elif (rating < 4 and option_a_personalized) or (rating > 4 and not option_a_personalized):
                    preferred_personalized.append('Preferred Personalized')
                else:
                    preferred_personalized.append('Preferred Default')
                
            df_clean[f"{meal_num}_preferred_personalized"] = preferred_personalized
    
    return df_clean

def plot_preference_distributions(df, output_dir='.'):
    """
    Plot distributions of preferences for each meal.
    
    Args:
        df (pd.DataFrame): DataFrame containing the survey responses
        output_dir (str): Directory to save plots
        
    Returns:
        None
    """
    # Check for preference columns
    pref_cols = [col for col in df.columns if 'preference' in col and not 'preferred' in col]
    
    if not pref_cols:
        print("No preference columns found in data.")
        return
    
    # Create figure for all meals
    plt.figure(figsize=(12, 8))
    
    # Create a bar chart for the distribution of preferences across all meals
    all_preferences = pd.Series(dtype='float64')
    for col in pref_cols:
        all_preferences = pd.concat([all_preferences, df[col].dropna()])
    
    counts = all_preferences.value_counts().sort_index()
    
    # Generate labels for the x-axis
    labels = {
        1: "Strongly prefer A",
        2: "Prefer A",
        3: "Somewhat prefer A",
        4: "Neutral",
        5: "Somewhat prefer B",
        6: "Prefer B",
        7: "Strongly prefer B"
    }
    
    counts.index = [labels.get(i, i) for i in counts.index]
    
    sns.barplot(x=counts.index, y=counts.values)
    plt.title('Distribution of Preference Ratings Across All Meals')
    plt.ylabel('Count')
    plt.xlabel('Preference Rating')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'preference_distribution.png'))
    plt.close()
    
    # Create individual meal preference distributions
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(pref_cols):
        plt.subplot(len(pref_cols), 1, i+1)
        meal_num = col.split('_')[0]
        
        # Convert to numeric to ensure proper ordering
        values = pd.to_numeric(df[col], errors='coerce')
        counts = values.value_counts().sort_index()
        counts.index = [labels.get(i, i) for i in counts.index]
        
        sns.barplot(x=counts.index, y=counts.values)
        plt.title(f'Preference Distribution for {meal_num.capitalize()}')
        plt.ylabel('Count')
        if i == len(pref_cols) - 1:  # Only add xlabel to bottom subplot
            plt.xlabel('Preference Rating')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'meal_preference_distributions.png'))
    plt.close()

def plot_personalization_preference(df, output_dir='.'):
    """
    Plot whether participants preferred the personalized option.
    
    Args:
        df (pd.DataFrame): DataFrame containing the survey responses with derived columns
        output_dir (str): Directory to save plots
        
    Returns:
        None
    """
    # Check for personalization preference columns
    pers_cols = [col for col in df.columns if 'preferred_personalized' in col]
    
    if not pers_cols:
        print("No personalization preference columns found in data.")
        return
    
    # Count overall preferences for personalization
    all_preferences = []
    all_meals = []
    
    for col in pers_cols:
        meal_num = col.split('_')[0]
        for pref in df[col].dropna():
            all_preferences.append(pref)
            all_meals.append(meal_num)
    
    # Create a DataFrame for easier plotting
    pref_df = pd.DataFrame({
        'Preference': all_preferences,
        'Meal': all_meals
    })
    
    # Plot overall preference for personalization
    plt.figure(figsize=(10, 6))
    counts = pref_df['Preference'].value_counts()
    sns.barplot(x=counts.index, y=counts.values)
    plt.title('Overall Preference for Personalization vs Default')
    plt.ylabel('Count')
    plt.xlabel('Preference')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'personalization_preference.png'))
    plt.close()
    
    # Plot preference for personalization by meal
    plt.figure(figsize=(12, 8))
    meal_pref = pref_df.groupby(['Meal', 'Preference']).size().unstack()
    
    # Fill NaN with 0 for plotting
    if meal_pref is not None and not meal_pref.empty:
        meal_pref = meal_pref.fillna(0)
        meal_pref.plot(kind='bar', stacked=False)
        plt.title('Preference for Personalization by Meal')
        plt.ylabel('Count')
        plt.xlabel('Meal')
        plt.legend(title='Preference')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'personalization_by_meal.png'))
        plt.close()

def plot_question_correlations(df, output_dir='.'):
    """
    Plot correlations between meal settings and personalization preference.
    
    Args:
        df (pd.DataFrame): DataFrame containing the survey responses
        output_dir (str): Directory to save plots
        
    Returns:
        None
    """
    # Check for relevant columns
    preference_cols = [col for col in df.columns if 'preferred_personalized' in col]
    
    if not preference_cols:
        print("No personalization preference columns found for correlation analysis.")
        return
    
    # Analyze meal setting impacts
    for setting in ['bite_order', 'ready_signal', 'verbal']:
        plt.figure(figsize=(14, 8))
        
        # Get all columns with this setting
        setting_cols = [col for col in df.columns if setting in col]
        
        # Create count plots for each meal
        n_meals = len(preference_cols)
        n_cols = 2
        n_rows = (n_meals + 1) // n_cols
        
        for i, (pref_col, setting_col) in enumerate(zip(preference_cols, setting_cols)):
            meal_num = pref_col.split('_')[0]
            plt.subplot(n_rows, n_cols, i+1)
            
            # Create a cross-tabulation
            if df[pref_col].notna().any() and df[setting_col].notna().any():
                crosstab = pd.crosstab(df[pref_col], df[setting_col])
                crosstab.plot(kind='bar', stacked=False)
                plt.title(f'{meal_num.capitalize()}: {setting.replace("_", " ").title()} vs Preference')
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.legend(title=setting.replace('_', ' ').title())
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{setting}_vs_preference.png'))
        plt.close()

def plot_meal_sequences(df, output_dir='.'):
    """
    Plot how preferences change across meals.
    
    Args:
        df (pd.DataFrame): DataFrame containing the survey responses
        output_dir (str): Directory to save plots
        
    Returns:
        None
    """
    # Get preference columns
    pref_cols = [col for col in df.columns if 'preference' in col and not 'preferred' in col]
    
    if len(pref_cols) < 2:  # Need at least 2 meals for sequences
        print("Not enough meals for sequence analysis.")
        return
    
    # Sort columns by meal number
    pref_cols.sort(key=lambda x: int(''.join(filter(str.isdigit, x.split('_')[0]))))
    
    # Create a line plot for each participant's preferences
    plt.figure(figsize=(12, 8))
    
    # Get meal numbers for x-axis
    meal_numbers = [int(''.join(filter(str.isdigit, col.split('_')[0]))) for col in pref_cols]
    
    # Plot each participant's preference trajectory
    for i, row in df.iterrows():
        values = [row[col] if pd.notna(row[col]) else None for col in pref_cols]
        
        # Only plot if we have at least two valid preferences
        if sum(1 for v in values if v is not None) >= 2:
            plt.plot(meal_numbers, values, marker='o', linestyle='-', alpha=0.5, 
                     label=f"Participant {row.get('response_id', i+1)}")
    
    # Adjust plot appearance
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.axhline(y=4, color='gray', linestyle='--', alpha=0.5, label='Neutral')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.title('Preference Ratings Across Meals')
    plt.xlabel('Meal Number')
    plt.ylabel('Preference Rating')
    plt.ylim(0.5, 7.5)
    
    # Custom legend that shows key information
    plt.legend(['Neutral (4)', 'Individual participants'], loc='center right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'preference_sequences.png'))
    plt.close()

def generate_all_plots(df, output_dir='.'):
    """
    Generate all plots for the data.
    
    Args:
        df (pd.DataFrame): DataFrame containing the survey responses
        output_dir (str): Directory to save plots
        
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the data
    df_clean = convert_preference_to_numeric(df)
    
    # Generate plots
    plot_preference_distributions(df_clean, output_dir)
    plot_personalization_preference(df_clean, output_dir)
    plot_question_correlations(df_clean, output_dir)
    plot_meal_sequences(df_clean, output_dir)
    
    print(f"All plots saved to {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Usage: python analyze_responses.py <response_csv_file> [output_directory]")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) == 3 else 'plots'
    
    # Load the data
    df = load_data(csv_path)
    
    if df is not None:
        generate_all_plots(df, output_dir)
    else:
        print("Failed to load data. Analysis aborted.") 