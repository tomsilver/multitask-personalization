"""Utility functions for analysis."""

import os
from pathlib import Path
from typing import Callable

import pandas as pd
from omegaconf import DictConfig, OmegaConf


def combine_results_csvs(
    directory: Path,
    results_filename: str = "eval_results.csv",
    config_filename: str = "config.yaml",
    config_fn: Callable[[DictConfig], bool] | None = None,
) -> pd.DataFrame:
    """Combine experimental results over runs into one dataframe."""
    combined_df = pd.DataFrame()
    for subdir in os.listdir(directory):
        subdir_path = directory / subdir
        if os.path.isdir(subdir_path) and subdir_path.name.isdigit():
            csv_path = subdir_path / results_filename
            config_path = subdir_path / config_filename
            cfg = OmegaConf.load(config_path)
            assert isinstance(cfg, DictConfig)
            if config_fn is not None and not config_fn(cfg):
                continue
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                assert subdir.isdigit()
                df["run"] = int(subdir)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df


def check_for_missing_results(df: pd.DataFrame) -> None:
    """Print warnings if any runs have fewer rows than other runs."""
    run_to_count: dict[int, int] = {}
    for run in sorted(set(df.run)):
        run_to_count[run] = sum(df.run == run)
    runs = sorted(run_to_count)
    print(f"Found {len(run_to_count)} runs in results: {runs}")
    max_count = max(run_to_count.values())
    for run, count in run_to_count.items():
        if count < max_count:
            print(f"WARNING: run {run} missing results ({count} < {max_count})")
