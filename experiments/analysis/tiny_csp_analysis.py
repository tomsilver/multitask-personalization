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
    keys = [
        k
        for k in df.keys()
        if k.startswith("eval_") and k.endswith("_user_satisfaction")
    ]
    df["mean_eval_user_satisfaction"] = df[keys].mean(axis=1)

    plt.subplots(1, 1)
    plt.title("CSP Approach in Tiny Env")
    sns.lineplot(
        data=df,
        x="training_step",
        y="mean_eval_user_satisfaction",
        estimator="mean",
        errorbar="sd",
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
