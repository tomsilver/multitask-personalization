"""Utility functions for analysis."""

import os
from pathlib import Path

import pandas as pd


def combine_csvs(
    directory: Path, results_filename: str = "results.csv"
) -> pd.DataFrame:
    """Combine experimental results over seeds into one dataframe."""
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame

    # Iterate over subdirectories in the given directory.
    for subdir in os.listdir(directory):
        subdir_path = directory / subdir
        # Check if the path is a directory and contains a results file.
        if os.path.isdir(subdir_path):
            csv_path = os.path.join(subdir_path, results_filename)
            if os.path.exists(csv_path):
                # Read the CSV file.
                df = pd.read_csv(csv_path)
                # Add the subdirectory name as a new column.
                assert subdir.isdigit()
                df["seed"] = int(subdir)
                # Append to the combined DataFrame
                combined_df = pd.concat([combined_df, df], ignore_index=True)

    return combined_df
