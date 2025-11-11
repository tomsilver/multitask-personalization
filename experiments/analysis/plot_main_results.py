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
    "cooking-nonstationary": "Cooking (Nonstationary)",
    # "cooking-stationary": "Cooking",
    # "cleaning-stationary": "Cleaning",
    # "overnight-stationary": "Books",
}

APPROACH_TO_DISPLAY_NAME = {
    # "ours": "CBTL (Ours)",
    # "ours_adapt_fancy": "CBTL (Fancy)",
    "ours_adapt_user_reset": "CBTL (User Reset)",
    "ours_adapt_periodic": "CBTL (Periodic Reset)",
    "ours_no_adapt": "CBTL (No Adapt)",
    "nothing_personal": "Free Explore",
    # "epsilon_greedy": "Epsilon Greedy",
    # "exploit_only": "Exploit Only",
    # "no_learning": "No Learning",
}

# Additional approach names
ADDITIONAL_APPROACH_TO_DISPLAY_NAME = {
    "ours": "CBTL (Ours)",
    "nothing_personal": "Free Explore",
    "epsilon_greedy": "Epsilon-Greedy",
    "exploit_only": "Exploit Only",
    "no_learning": "No Personalization",
}

COLOR_PALETTE = [
    "#66c2a5",  # soft green-teal
    "#fc8d62",  # soft orange
    "#8da0cb",  # periwinkle blue
    "#e78ac3",  # soft magenta
    "#a6d854",  # lime green
    "#ffd92f",  # sunflower yellow
    "#e5c494",  # sand brown
    "#b3b3b3",  # light gray
    "#bebada",  # lavender
    "#fb8072",  # coral
]

# Additional color definitions
c = lambda x : x/255.0

colors = [
        (c(30), c(60), c(250)),
        (c(21), c(151), c(165)),  #0,
        (c(230), c(111), c(81)), #2
        (c(155), c(89), c(182)),
        (c(254), c(183), c(5)),  #1
        ]

# https://colorbrewer2.org/#type=diverging&scheme=Spectral&n=8
APPROACH_TO_COLOR = {
    # "ours": "#3288bd",
    # assign different colors to each approach
    "ours_no_adapt": "#aa3377",
    "ours_adapt_periodic": COLOR_PALETTE[2],
    # "ours_adapt_fancy": COLOR_PALETTE[2],
    "ours_adapt_user_reset": COLOR_PALETTE[9],
    "nothing_personal": "#4d9221",
    # "epsilon_greedy": "#abdda4",
    # "exploit_only": "#e6f598",
    # "no_learning": "#fee08b",
}

# Additional color mappings
ADDITIONAL_APPROACH_TO_COLOR = {
    "ours": "#aa3377",
    "nothing_personal": "#4d9221",
    "epsilon_greedy": "#bf812d",
    "exploit_only": "#4477aa",
    "no_learning": "#66ccee",
}

# Colors for preference shift backgrounds
SHIFT_COLORS = [
    "#C6E2FF",  # Pale blue
    "#FFB6C1",  # Light pink
    "#98FB98",  # Pale green
    "#DDA0DD",  # Plum
    "#F0E68C",  # Khaki
    "#E6E6FA",  # Lavender
    "#FFA07A",  # Light salmon
    "#B0E0E6",  # Powder blue
    "#D8BFD8",  # Thistle
    "#FFDAB9",  # Peach puff
]


def _create_config_fn(
    env_name: str, approach_name: str
) -> Callable[[DictConfig], bool]:
    def _fn(cfg: DictConfig) -> bool:
        return cfg.env_name == env_name and cfg.approach_name == approach_name

    return _fn


def _get_shift_times(results_dir: Path, env_name: str, approach_name: str) -> list[int]:
    """Get the times when preference shifts occurred from the results files."""
    config_fn = _create_config_fn(env_name, approach_name)
    df = combine_results_csvs(results_dir, config_fn=config_fn)
    if df.empty:
        return []

    # Get all times where a shift occurred
    shift_df = df[
        df["preference_shift"] == True  # pylint: disable=singleton-comparison
    ]
    print(f"shift_df: {shift_df}")
    if shift_df.empty:
        return []

    # Get the evaluation times
    eval_times = df["training_execution_time"].tolist()

    # For each shift, find the previous evaluation time
    shift_times = []
    for shift_time in shift_df["training_execution_time"]:
        # Find the index of the evaluation time that's just before the shift
        prev_eval_idx = max(0, eval_times.index(shift_time) - 1)
        shift_times.append(eval_times[prev_eval_idx])

    return sorted(set(shift_times))  # Ensure times are in order


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
        ax.set_ylim((-1.05, 1.05))

        # Colored background sections for preference shifts
        if env_name == "cooking-nonstationary":
            # Get shift times from the first approach's results
            first_approach = list(APPROACH_TO_DISPLAY_NAME.keys())[0]
            shift_times = _get_shift_times(results_dir, env_name, first_approach)

            if shift_times:
                # Get the maximum execution time from the data
                df = combine_results_csvs(
                    results_dir, config_fn=_create_config_fn(env_name, first_approach)
                )
                max_time = df["training_execution_time"].max()

                # Create sections between shifts
                section_starts = [0] + shift_times
                section_ends = shift_times + [max_time]

                print(f"section_starts: {section_starts}")
                print(f"section_ends: {section_ends}")
                # Add colored background sections
                for j, (start, end) in enumerate(zip(section_starts, section_ends)):
                    print(f"start: {start}, end: {end}")
                    ax.axvspan(
                        start, end, alpha=0.3, color=SHIFT_COLORS[j % len(SHIFT_COLORS)]
                    )
                # Add vertical lines for shift times
                for shift_time in shift_times:
                    ax.axvline(x=shift_time, color="gray", linestyle="--", alpha=0.5)

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
                # lines.append(line.get_lines()[-1])
                # Create a separate line for the legend with increased thickness
                legend_line = plt.Line2D([0], [0], color=color, linewidth=5.0)
                lines.append(legend_line)
                labels.append(approach_display_name)

        if i == 0:
            ax.set_ylabel("Simulated User Satisfaction")
        else:
            ax.set_ylabel("")

    # Place a single shared legend to the right of the subplots.
    # fig.legend(lines, labels, loc="center right", bbox_to_anchor=(1.0, 0.5))
    
    # Alternative legend placement below subplots
    # fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, 0.05), ncol=len(APPROACH_TO_DISPLAY_NAME), 
    #           prop={'size': 10})
    fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, -0.0), ncol=len(APPROACH_TO_DISPLAY_NAME)//2, 
              prop={'size': 15})

    # Adjust layout with extra space for legend.
    plt.tight_layout(rect=(0, 0.1, 1.0, 1.0))
    # Alternative layout without rect
    # plt.tight_layout()

    # plt.savefig(outfile, dpi=1000, bbox_inches="tight", pad_inches=0.05)
    # Alternative save with transparency
    plt.savefig(outfile, dpi=1000, bbox_inches="tight", pad_inches=0.05, transparent=True)
    print(f"Wrote out to {outfile}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("outfile", type=Path)
    args = parser.parse_args()
    _main(args.results_dir, args.outfile)
