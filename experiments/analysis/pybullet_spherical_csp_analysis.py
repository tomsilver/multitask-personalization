"""Analyze results for running the CSP approach in the PyBullet with the
spherical ROM model e.g.,

```
python experiments/run_single_experiment.py -m seed=1,2,3,4,5,6,7,8,9,10 \
    approach=csp env=pybullet
```
"""

import argparse
from pathlib import Path

import seaborn as sns
from analysis_utils import combine_results_csvs
from matplotlib import pyplot as plt


def _main(results_dir: Path, outfile: Path) -> None:
    plt.rcParams.update({"font.size": 16})

    df = combine_results_csvs(results_dir)

    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    ax0, ax1 = axes  # type: ignore
    fig.suptitle("CSP Approach in PyBullet Env with Spherical ROM")

    # Make a plot showing returns over time.
    ax0.set_title("Returns")
    sns.regplot(
        df,
        x="episode",
        y="returns",
        order=15,
        scatter_kws={
            "s": 2,
            "color": (0, 0, 1, 0.1),
        },
        line_kws={
            "lw": 3,
            "color": "red",
        },
        ax=ax0,
    )

    # Make a plot showing learned proximity over time.
    ax1.set_title("Learned Radius")
    sns.regplot(
        df,
        x="episode",
        y="spherical_rom_radius",
        order=15,
        scatter_kws={
            "s": 2,
            "color": (0, 0, 1, 0.1),
        },
        line_kws={
            "lw": 3,
            "color": "red",
        },
        ax=ax1,
    )

    plt.tight_layout()
    plt.savefig(outfile, dpi=500)
    print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("outfile", type=Path)
    args = parser.parse_args()
    _main(args.results_dir, args.outfile)
