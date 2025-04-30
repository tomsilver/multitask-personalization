#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Use custom style
plt.style.use('custom.mplstyle')

def load_and_prepare_data():
    """Load both CSV files and prepare the data for plotting."""
    # Load the raw data
    personalized_df = pd.read_csv('raw_predictions.csv')
    non_personalized_df = pd.read_csv('raw_predictions__non_personalized.csv')
    
    # Exclude occlusion predictions
    personalized_df = personalized_df[personalized_df['prediction_type'] != 'occlusion']
    non_personalized_df = non_personalized_df[non_personalized_df['prediction_type'] != 'occlusion']
    
    # Group by meal number and calculate mean success rate and SEM
    def aggregate_data(df):
        # First aggregate by prediction type and meal number
        type_meal_agg = df.groupby(['prediction_type', 'meal_number']).agg({
            'success': ['mean', 'std', 'count']
        }).reset_index()
        
        # Flatten the multi-level columns
        type_meal_agg.columns = ['prediction_type', 'meal_number', 'mean', 'std', 'count']
        
        # Calculate SEM for each prediction type and meal
        type_meal_agg['sem'] = type_meal_agg['std'] / np.sqrt(type_meal_agg['count'])
        
        # Then aggregate across prediction types
        # For the mean, we take the average of means
        # For the SEM, we need to combine the variances and then take the square root
        meal_agg = type_meal_agg.groupby('meal_number').agg({
            'mean': 'mean',
            'sem': lambda x: np.sqrt(np.sum(x**2))  # Combine SEMs by adding variances
        }).reset_index()
        
        # Rename columns for clarity
        meal_agg.columns = ['meal_number', 'mean_success_rate', 'sem']
        return meal_agg
    
    personalized_agg = aggregate_data(personalized_df)
    non_personalized_agg = aggregate_data(non_personalized_df)
    
    return personalized_agg, non_personalized_agg

def create_comparison_plot():
    """Create a line plot comparing personalized vs non-personalized predictions."""
    personalized_agg, non_personalized_agg = load_and_prepare_data()
    
    plt.figure(figsize=(5, 6))
    
    # Plot lines with error bars
    plt.errorbar(personalized_agg['meal_number'], 
                personalized_agg['mean_success_rate'],
                yerr=personalized_agg['sem'],
                label='CBTL (Ours)',
                marker='o',
                capsize=2,
                color="#aa3377")
    
    plt.errorbar(non_personalized_agg['meal_number'],
                non_personalized_agg['mean_success_rate'],
                yerr=non_personalized_agg['sem'],
                label='No Personalization',
                marker='s',
                capsize=2,
                color="#66ccee")
    
    # Customize the plot
    plt.xlabel('Meal Number')
    plt.ylabel('Mean Prediction Success Rate')
    plt.legend(labelspacing=0.9, loc='lower right', prop={'size': 16}, markerscale=0.7)  # Smaller markers in legend
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to only show actual meal numbers
    plt.xticks(personalized_agg['meal_number'])
    
    # Set y-axis limits to show full range of possible values
    plt.ylim(0, 1)
    
    # Save the plot
    plt.savefig('prediction_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_comparison_plot() 
