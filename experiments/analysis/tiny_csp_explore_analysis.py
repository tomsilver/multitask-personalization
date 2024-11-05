"""Analyze different explore methods in TinyCSP.

```
python experiments/run_single_experiment.py -m +experiment=tiny_csp \
    approach.explore_method=neighborhood,nothing-personal,ensemble \
    seed="range(1, 11)"
```
"""

import argparse
from pathlib import Path

import seaborn as sns
from analysis_utils import combine_results_csvs
from matplotlib import pyplot as plt


def _main(results_dir: Path, outfile: Path) -> None:
    plt.rcParams.update({"font.size": 12})

    explore_method_to_dfs = {
        "neighborhood": combine_results_csvs(
            results_dir,
            config_fn=lambda c: c.approach.explore_method == "neighborhood",
        ),
        "nothing-personal": combine_results_csvs(
            results_dir,
            config_fn=lambda c: c.approach.explore_method == "nothing-personal",
        ),
        "ensemble": combine_results_csvs(
            results_dir, config_fn=lambda c: c.approach.explore_method == "ensemble"
        ),
    }

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    fig.suptitle("Comparing Exploration Methods for CSP Approach in Tiny Env")

    for explore_method, df in explore_method_to_dfs.items():
        df = df[~df.explore]
        sns.regplot(
            df,
            x="episode",
            y="returns",
            order=15,
            scatter=False,
            line_kws={
                "lw": 3,
            },
            ax=ax,
            label=explore_method,
        )

    ax.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=500)
    print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("outfile", type=Path)
    args = parser.parse_args()
    _main(args.results_dir, args.outfile)
