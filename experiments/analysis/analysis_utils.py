"""Utility functions for analysis."""

import os
from pathlib import Path

import pandas as pd


def combine_results_csvs(
    directory: Path, results_filename: str = "results.csv"
) -> pd.DataFrame:
    """Combine experimental results over seeds into one dataframe."""
    combined_df = pd.DataFrame()
    for subdir in os.listdir(directory):
        subdir_path = directory / subdir
        if os.path.isdir(subdir_path):
            csv_path = os.path.join(subdir_path, results_filename)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                assert subdir.isdigit()
                df["seed"] = int(subdir)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df
