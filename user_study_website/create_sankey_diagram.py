#!/usr/bin/env python3
import json
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import argparse
from collections import defaultdict
import random

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

def extract_choices(responses, choice_type):
    """
    Extract choices from responses and create a DataFrame for Sankey diagram.
    
    Args:
        responses: List of parsed JSON responses
        choice_type: Type of choice to extract ('dip', 'be_verbal', or 'ready_signal')
        
    Returns:
        DataFrame with columns: source, target, value
    """
    # First, collect all unique paths and their counts
    path_counts = defaultdict(int)
    
    for response in responses:
        choice_sequence = []
        for answer in response['answers']:
            if choice_type == 'dip':
                bite_order = answer['bite_order']['value']
                if 'dipped in' in bite_order:
                    choice = bite_order.split('dipped in ')[1]
                else:
                    choice = 'no dip'
            elif choice_type == 'be_verbal':
                choice = str(answer['verbal']['value']).lower()
            elif choice_type == 'ready_signal':
                choice = answer['ready_signal']['value'].lower()
            choice_sequence.append(choice)
        
        # Convert sequence to string for counting
        sequence_str = ' -> '.join(choice_sequence)
        path_counts[sequence_str] += 1
    
    # Now create the Sankey diagram data
    sources = []
    targets = []
    values = []
    
    # Process each unique path
    for sequence_str, count in path_counts.items():
        choice_sequence = sequence_str.split(' -> ')
        
        # Create links between consecutive choices
        for i in range(len(choice_sequence) - 1):
            # Use the sequence up to the current meal as the source
            source_sequence = ' -> '.join(choice_sequence[:i+1])
            source = f"Meal {i+1}: {choice_sequence[i]}_{source_sequence}"
            
            # Use the sequence up to the next meal as the target
            target_sequence = ' -> '.join(choice_sequence[:i+2])
            target = f"Meal {i+2}: {choice_sequence[i+1]}_{target_sequence}"
            
            sources.append(source)
            targets.append(target)
            values.append(count)
    
    # Create DataFrame
    df = pd.DataFrame({
        'source': sources,
        'target': targets,
        'value': values
    })
    
    return df

def get_color_scheme(choice_type):
    """
    Get the color scheme for the specified choice type.
    
    Args:
        choice_type: Type of choice ('dip', 'be_verbal', or 'ready_signal')
        
    Returns:
        Dictionary mapping choices to colors
    """
    if choice_type == 'dip':
        return {
            'ketchup': 'rgba(255, 0, 0, 0.6)',      # Red
            'hummus': 'rgba(210, 180, 140, 0.6)',   # Tan
            'ranch': 'rgba(0, 255, 0, 0.6)',        # Green
            'bbq': 'rgba(139, 69, 19, 0.6)',        # Brown
            'no dip': 'rgba(128, 128, 128, 0.6)'    # Gray
        }
    elif choice_type == 'be_verbal':
        return {
            'true': 'rgba(100, 149, 237, 0.6)',     # Cornflower Blue
            'false': 'rgba(176, 196, 222, 0.6)',    # Light Steel Blue
        }
    elif choice_type == 'ready_signal':
        return {
            'ready': 'rgba(0, 128, 0, 0.6)',        # Green
            'not ready': 'rgba(255, 0, 0, 0.6)',    # Red
        }
    return {}

def create_sankey_diagram(df, output_file, choice_type):
    """
    Create and save a Sankey diagram.
    
    Args:
        df: DataFrame with source, target, and value columns
        output_file: Path to save the HTML file
        choice_type: Type of choice being visualized
    """
    # Get unique nodes
    nodes = list(set(df['source'].unique()) | set(df['target'].unique()))
    
    # Create node indices
    node_indices = {node: i for i, node in enumerate(nodes)}
    
    # Create color mapping for different choices
    unique_choices = set()
    for node in nodes:
        choice = node.split(': ')[1].split('_')[0]  # Extract choice name from node label
        unique_choices.add(choice)
    
    # Get base color scheme
    choice_colors = get_color_scheme(choice_type)
    
    # Generate random colors for any other choices
    for choice in unique_choices:
        if choice not in choice_colors:
            r = random.randint(50, 200)
            g = random.randint(50, 200)
            b = random.randint(50, 200)
            choice_colors[choice] = f"rgba({r}, {g}, {b}, 0.6)"
    
    # Create node colors
    node_colors = []
    for node in nodes:
        if node.startswith('Meal 1:'):  # Initial nodes
            node_colors.append('rgba(200, 200, 200, 0.6)')  # Light gray for initial nodes
        else:
            choice = node.split(': ')[1].split('_')[0]
            node_colors.append(choice_colors[choice])
    
    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[node.split(': ')[1].split('_')[0] for node in nodes],  # Show only choice names
            color=node_colors
        ),
        link=dict(
            source=[node_indices[source] for source in df['source']],
            target=[node_indices[target] for target in df['target']],
            value=df['value'],
            color=[choice_colors[source.split(': ')[1].split('_')[0]] for source in df['source']]  # Color by source choice
        )
    )])
    
    # Update layout
    title_map = {
        'dip': 'Dip Choices Flow Across Meals',
        'be_verbal': 'Verbal Preference Flow Across Meals',
        'ready_signal': 'Ready Signal Flow Across Meals'
    }
    fig.update_layout(
        title_text=title_map.get(choice_type, 'Choices Flow Across Meals'),
        font_size=24,
        height=800
    )
    
    # Save the figure
    fig.write_html(output_file)

def main(file_path, output_file, choice_type):
    # Load responses
    responses = load_responses(file_path)
    
    # Extract choices and create DataFrame
    df = extract_choices(responses, choice_type)
    
    # Create and save Sankey diagram
    create_sankey_diagram(df, output_file, choice_type)
    print(f"Sankey diagram saved to {output_file}")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Create a Sankey diagram of choices across meals.')
    parser.add_argument('file_path', nargs='?', default='user_data_example.txt',
                        help='Path to the user data file (default: user_data_example.txt)')
    parser.add_argument('--output', '-o', default='choices_sankey.html',
                        help='Output HTML file path (default: choices_sankey.html)')
    parser.add_argument('--type', '-t', choices=['dip', 'be_verbal', 'ready_signal'],
                        default='dip', help='Type of choice to visualize (default: dip)')
    
    # Parse arguments
    args = parser.parse_args()
    file_path = Path(args.file_path)
    
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist.")
    else:
        main(file_path, args.output, args.type) 