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

def extract_dip_choices(responses):
    """
    Extract dip choices from responses and create a DataFrame for Sankey diagram.
    
    Args:
        responses: List of parsed JSON responses
        
    Returns:
        DataFrame with columns: source, target, value
    """
    # First, collect all unique paths and their counts
    path_counts = defaultdict(int)
    
    for response in responses:
        dip_sequence = []
        for answer in response['answers']:
            bite_order = answer['bite_order']['value']
            if 'dipped in' in bite_order:
                dip = bite_order.split('dipped in ')[1]
            else:
                dip = 'no dip'
            dip_sequence.append(dip)
        
        # Convert sequence to string for counting
        sequence_str = ' -> '.join(dip_sequence)
        path_counts[sequence_str] += 1
    
    # Now create the Sankey diagram data
    sources = []
    targets = []
    values = []
    
    # Process each unique path
    for sequence_str, count in path_counts.items():
        dip_sequence = sequence_str.split(' -> ')
        
        # Create links between consecutive dips
        for i in range(len(dip_sequence) - 1):
            # Use the sequence up to the current meal as the source
            source_sequence = ' -> '.join(dip_sequence[:i+1])
            source = f"Meal {i+1}: {dip_sequence[i]}_{source_sequence}"
            
            # Use the sequence up to the next meal as the target
            target_sequence = ' -> '.join(dip_sequence[:i+2])
            target = f"Meal {i+2}: {dip_sequence[i+1]}_{target_sequence}"
            
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

def create_sankey_diagram(df, output_file):
    """
    Create and save a Sankey diagram.
    
    Args:
        df: DataFrame with source, target, and value columns
        output_file: Path to save the HTML file
    """
    # Get unique nodes
    nodes = list(set(df['source'].unique()) | set(df['target'].unique()))
    
    # Create node indices
    node_indices = {node: i for i, node in enumerate(nodes)}
    
    # Create color mapping for different paths
    unique_paths = set()
    for node in nodes:
        if '_' in node:  # Only consider nodes with path information
            path = node.split('_', 1)[1]
            unique_paths.add(path)
    
    path_colors = {}
    for path in unique_paths:
        # Generate a random color with good visibility
        r = random.randint(50, 200)
        g = random.randint(50, 200)
        b = random.randint(50, 200)
        path_colors[path] = f"rgba({r}, {g}, {b}, 0.6)"
    
    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[node.split('_')[0] for node in nodes],  # Remove path ID from labels
            color="lightblue"
        ),
        link=dict(
            source=[node_indices[source] for source in df['source']],
            target=[node_indices[target] for target in df['target']],
            value=df['value'],
            color=[path_colors[target.split('_', 1)[1]] for target in df['target']]  # Color by path
        )
    )])
    
    # Update layout
    fig.update_layout(
        title_text="Dip Choices Flow Across Meals (Branching Paths)",
        font_size=10,
        height=800
    )
    
    # Save the figure
    fig.write_html(output_file)

def main(file_path, output_file):
    # Load responses
    responses = load_responses(file_path)
    
    # Extract dip choices and create DataFrame
    df = extract_dip_choices(responses)
    
    # Create and save Sankey diagram
    create_sankey_diagram(df, output_file)
    print(f"Sankey diagram saved to {output_file}")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Create a Sankey diagram of dip choices across meals.')
    parser.add_argument('file_path', nargs='?', default='user_data_example.txt',
                        help='Path to the user data file (default: user_data_example.txt)')
    parser.add_argument('--output', '-o', default='dip_choices_sankey.html',
                        help='Output HTML file path (default: dip_choices_sankey.html)')
    
    # Parse arguments
    args = parser.parse_args()
    file_path = Path(args.file_path)
    
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist.")
    else:
        main(file_path, args.output) 