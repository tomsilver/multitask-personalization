"""Analyze results for running the CSP approach in the TinyEnv, e.g.,

python experiments/run_single_experiment.py -m  approach=csp env=tiny \
seed=1,2,3,4,5,6,7,8,9,10
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
    fig.suptitle("CSP Approach in Tiny Env")

    # Make a plot showing returns over time.
    ax0.set_title("Returns")
    for epsilon, color in zip(df.epsilon.unique(), ["red", "blue"], strict=True):
        sns.regplot(
            df[df.epsilon == epsilon],
            x="episode",
            y="returns",
            order=15,
            scatter_kws={
                "s": 2,
                "color": (0, 0, 1, 0.1),
            },
            line_kws={
                "lw": 3,
                "color": color,
            },
            scatter=False,
            ax=ax0,
            label=f"Epsilon={epsilon}",
        )
    ax0.legend()

    # Make a plot showing learned proximity over time.
    ax1.set_title("Learned Proximity")
    for epsilon, color in zip(df.epsilon.unique(), ["red", "blue"], strict=True):
        sns.regplot(
            df[df.epsilon == epsilon],
            x="episode",
            y="tiny_user_proximity_learned_distance",
            order=15,
            scatter_kws={
                "s": 2,
                "color": (0, 0, 1, 0.1),
            },
            line_kws={
                "lw": 3,
                "color": color,
            },
            ax=ax1,
        )
    ax1.legend()

    plt.tight_layout()
    plt.savefig(outfile, dpi=500)
    print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("outfile", type=Path)
    args = parser.parse_args()
    _main(args.results_dir, args.outfile)
