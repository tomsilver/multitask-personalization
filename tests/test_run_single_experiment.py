"""Tests that main experiments run without crashing (for a very short time)."""

import subprocess
import sys
from pathlib import Path


def test_run_single_experiment():
    """Tests for run_single_experiment.py."""
    # Get the path to the experiment file.
    filepath = (
        Path(__file__).resolve().parent.parent
        / "experiments"
        / "run_single_experiment.py"
    )

    # Ensure the script exists.
    assert filepath.is_file(), f"{filepath} does not exist"

    # Set up the flags.
    flags = [
        "-m",
        "env=tiny",
        "approach=ours",
        "seed=123",
        "env.max_environment_steps=10",
        "env.eval_frequency=5",
        "env.num_eval_trials=1",
        "csp_solver=random_walk",
        "csp_solver.min_num_satisfying_solutions=1",
        "csp_solver.max_iters=1000",
    ]

    # Run the script using subprocess.
    result = subprocess.run(
        [sys.executable, str(filepath)] + flags,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    # Check the return code to ensure the script executed successfully.
    assert result.returncode == 0, (
        f"Script crashed with return code {result.returncode}. "
        f"Stderr: {result.stderr.decode('utf-8')}"
    )
