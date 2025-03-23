"""Create main results plots."""

import argparse
from pathlib import Path
from typing import Callable

import seaborn as sns
from analysis_utils import check_for_missing_results, combine_results_csvs
from matplotlib import pyplot as plt
from omegaconf import DictConfig

ENV_TO_DISPLAY_NAME = {
    # "tiny": "Tiny",
    # "cooking-stationary": "Cooking",
    "overnight-stationary": "Overnight Assistance",
}

APPROACH_TO_DISPLAY_NAME = {
    "ours": "CBTL (Ours)",
    "nothing_personal": "Free Explore",
    "epsilon_greedy": "Epsilon Greedy",
    "exploit_only": "Exploit Only",
    "no_learning": "No Learning",
}

# https://colorbrewer2.org/#type=diverging&scheme=Spectral&n=8
APPROACH_TO_COLOR = {
    "ours": "#3288bd",
    "nothing_personal": "#66c2a5",
    "epsilon_greedy": "#abdda4",
    "exploit_only": "#e6f598",
    "no_learning": "#fee08b",
}


def _create_config_fn(
    env_name: str, approach_name: str
) -> Callable[[DictConfig], bool]:

    def _fn(cfg: DictConfig) -> bool:
        return cfg.env_name == env_name and cfg.approach_name == approach_name

    return _fn


def _main(results_dir: Path, outfile: Path) -> None:
    plt.style.use(Path(__file__).parent / "custom.mplstyle")

    num_envs = len(ENV_TO_DISPLAY_NAME)
    fig, axes = plt.subplots(
        1, len(ENV_TO_DISPLAY_NAME), figsize=(6 * num_envs, 5), squeeze=False
    )

    lines = []  # To collect line handles for legend
    labels = []  # To collect labels for legend

    for i, (ax, (env_name, env_display_name)) in enumerate(
        zip(axes[0], ENV_TO_DISPLAY_NAME.items())
    ):
        ax.set_title(env_display_name)
        ax.set_xlabel("Simulated Execution Time")

        for approach_name, approach_display_name in APPROACH_TO_DISPLAY_NAME.items():
            print(f"Combining results for {env_name}, {approach_name}")
            color = APPROACH_TO_COLOR[approach_name]
            config_fn = _create_config_fn(env_name, approach_name)
            df = combine_results_csvs(results_dir, config_fn=config_fn)
            if df.empty:
                print(f"WARNING: no data found for {env_name}: {approach_name}")
                continue
            check_for_missing_results(df)
            line = sns.lineplot(
                data=df,
                x="training_execution_time",
                y="eval_mean_user_satisfaction",
                estimator="mean",
                errorbar="se",
                ax=ax,
                color=color,
                label=None,
            )

            # Only add to legend collection for the first subplot.
            if env_name == list(ENV_TO_DISPLAY_NAME.keys())[0]:
                lines.append(line.get_lines()[-1])
                labels.append(approach_display_name)

        if i == 0:
            ax.set_ylabel("Simulated User Satisfaction")
        else:
            ax.set_ylabel("")

    # Place a single shared legend to the right of the subplots.
    fig.legend(lines, labels, loc="center right", bbox_to_anchor=(1.15, 0.5))

    # Adjust layout with extra space for legend.
    plt.tight_layout(rect=(0, 0, 0.85, 1))

    plt.savefig(outfile, dpi=500, bbox_inches="tight")
    print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("outfile", type=Path)
    args = parser.parse_args()
    _main(args.results_dir, args.outfile)
