"""Analyze results for running the CSP approach in the TinyEnv, e.g.,

```
python experiments/run_single_experiment.py -m +experiment=tiny_csp \
    seed="range(1, 11)"
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

    # Subselect non-explore steps.
    df = df[~df.user_allows_explore]

    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    ax0, ax1 = axes  # type: ignore
    fig.suptitle("CSP Approach in Tiny Env")

    # Make a plot showing user satisfaction over time.
    ax0.set_title("User Satisfaction")
    sns.regplot(
        df,
        x="step",
        y="user_satisfaction",
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
    ax1.set_title("Learned Proximity")
    sns.regplot(
        df,
        x="step",
        y="tiny_user_proximity_learned_distance",
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
