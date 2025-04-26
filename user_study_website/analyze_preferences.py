import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
from pathlib import Path
from decode_google_form_data import process_google_form_data

def setup_plotting_style():
    """Set up the plotting style for consistent visualizations"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("Set2")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

def create_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def analyze_preferences(df_answers, output_dir=None):
    """
    Analyze meal preferences and generate visualizations
    
    Args:
        df_answers: DataFrame with meal preference data
        output_dir: Directory to save visualizations (optional)
    """
    if df_answers.empty:
        print("No preference data available for analysis")
        return
    
    # Set up plotting style
    setup_plotting_style()
    
    # Create output directory if specified
    if output_dir:
        output_path = create_output_dir(output_dir)
    
    print(f"\nAnalyzing data for {len(df_answers)} meals...")
    
    # 1. Preference Rating Distribution
    plt.figure(figsize=(10, 6))
    if 'preference_rating' in df_answers.columns:
        sns.countplot(x='preference_rating', data=df_answers, 
                      order=sorted(df_answers['preference_rating'].unique()))
        plt.title('Distribution of Preference Ratings')
        plt.xlabel('Preference Rating (1=Strongly prefer A, 7=Strongly prefer B)')
        plt.ylabel('Count')
        
        if output_dir:
            plt.savefig(output_path / 'preference_ratings_distribution.png', dpi=300, bbox_inches='tight')
            print(f"Saved preference ratings distribution plot to {output_path / 'preference_ratings_distribution.png'}")
        plt.show()
    
    # 2. Preference for Personalized vs Default
    if 'preferred_option' in df_answers.columns:
        plt.figure(figsize=(8, 6))
        preference_counts = df_answers['preferred_option'].value_counts()
        plt.pie(preference_counts, labels=preference_counts.index, autopct='%1.1f%%', 
                startangle=90, shadow=False)
        plt.axis('equal')
        plt.title('Preference for Personalized vs Default Options')
        
        if output_dir:
            plt.savefig(output_path / 'personalized_vs_default_preference.png', dpi=300, bbox_inches='tight')
            print(f"Saved personalized vs default preference plot to {output_path / 'personalized_vs_default_preference.png'}")
        plt.show()
    
    # 3. Preference Evolution Over Meals
    if 'meal_number' in df_answers.columns and 'preferred_option' in df_answers.columns:
        plt.figure(figsize=(10, 6))
        preference_by_meal = pd.crosstab(df_answers['meal_number'], df_answers['preferred_option'], normalize='index')
        preference_by_meal.plot(kind='bar', stacked=True)
        plt.title('Evolution of Preferences Across Meals')
        plt.xlabel('Meal Number')
        plt.ylabel('Proportion')
        plt.xticks(rotation=0)
        plt.legend(title='Preference')
        
        if output_dir:
            plt.savefig(output_path / 'preference_evolution.png', dpi=300, bbox_inches='tight')
            print(f"Saved preference evolution plot to {output_path / 'preference_evolution.png'}")
        plt.show()
    
    # 4. Specific Preferences Analysis - Bite Order
    if 'bite_order' in df_answers.columns:
        plt.figure(figsize=(10, 6))
        bite_order_counts = df_answers['bite_order'].value_counts()
        bite_order_counts.plot(kind='bar')
        plt.title('Preferred Bite Order')
        plt.xlabel('Bite Order')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        if output_dir:
            plt.savefig(output_path / 'bite_order_preference.png', dpi=300, bbox_inches='tight')
            print(f"Saved bite order preference plot to {output_path / 'bite_order_preference.png'}")
        plt.show()
    
    # 5. Specific Preferences Analysis - Ready Signal
    if 'ready_signal' in df_answers.columns:
        plt.figure(figsize=(10, 6))
        ready_signal_counts = df_answers['ready_signal'].value_counts()
        ready_signal_counts.plot(kind='bar')
        plt.title('Preferred Ready Signal')
        plt.xlabel('Ready Signal')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        if output_dir:
            plt.savefig(output_path / 'ready_signal_preference.png', dpi=300, bbox_inches='tight')
            print(f"Saved ready signal preference plot to {output_path / 'ready_signal_preference.png'}")
        plt.show()
    
    # 6. Specific Preferences Analysis - Verbal
    if 'verbal' in df_answers.columns:
        plt.figure(figsize=(8, 6))
        verbal_counts = df_answers['verbal'].value_counts()
        plt.pie(verbal_counts, labels=verbal_counts.index, autopct='%1.1f%%',
                startangle=90, shadow=False)
        plt.axis('equal')
        plt.title('Preference for Robot Verbality')
        
        if output_dir:
            plt.savefig(output_path / 'verbal_preference.png', dpi=300, bbox_inches='tight')
            print(f"Saved verbal preference plot to {output_path / 'verbal_preference.png'}")
        plt.show()
    
    # 7. Correlation Matrix (if enough variables)
    if len(df_answers) > 3 and len(df_answers.columns) > 4:
        # Convert string columns to categorical for correlation analysis
        df_corr = df_answers.copy()
        for col in df_corr.select_dtypes(include=['object']).columns:
            df_corr[col] = pd.Categorical(df_corr[col]).codes
        
        # Calculate correlations (drop NaN values)
        corr = df_corr.corr(method='spearman', numeric_only=True)
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                   vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
        plt.title('Correlation Matrix of User Preferences')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_path / 'preference_correlations.png', dpi=300, bbox_inches='tight')
            print(f"Saved preference correlations plot to {output_path / 'preference_correlations.png'}")
        plt.show()
    
    # 8. Summary Statistics
    print("\n=== Preference Summary Statistics ===")
    
    if 'preferred_option' in df_answers.columns:
        print("\nOverall Preference Distribution:")
        print(df_answers['preferred_option'].value_counts(normalize=True).apply(lambda x: f"{x:.1%}"))
    
    # Common patterns in preferences
    print("\nMost Common Preferences:")
    common_prefs = {}
    for col in ['bite_order', 'ready_signal', 'verbal']:
        if col in df_answers.columns:
            common_prefs[col] = df_answers[col].value_counts().index[0]
    
    for pref, value in common_prefs.items():
        print(f"  - {pref.replace('_', ' ').title()}: {value}")

def analyze_intake_data(df_intake):
    """
    Analyze participant intake data
    
    Args:
        df_intake: DataFrame with participant intake data
    """
    if df_intake.empty:
        print("No intake data available for analysis")
        return
    
    print("\n=== Participant Information ===")
    
    for column in df_intake.columns:
        if column == 'name':
            continue  # Skip name for privacy
        print(f"{column.replace('_', ' ').title()}: {df_intake.iloc[0][column]}")

def main():
    parser = argparse.ArgumentParser(description='Analyze Google Form preference data')
    parser.add_argument('--data', type=str, help='Base64-encoded data from Google Form')
    parser.add_argument('--file', type=str, help='Path to file containing Base64-encoded data')
    parser.add_argument('--output', type=str, help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Get data from either command line argument or file
    compressed_data = None
    if args.data:
        compressed_data = args.data
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                compressed_data = f.read().strip()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    else:
        print("Please provide either --data or --file argument")
        return
    
    # Process the data using our decoder module
    df_answers, df_intake = process_google_form_data(compressed_data)
    
    # Analyze the data
    analyze_preferences(df_answers, args.output)
    analyze_intake_data(df_intake)

if __name__ == "__main__":
    main() 