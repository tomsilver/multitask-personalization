"""Create real robot results plots."""

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from plot_main_results import APPROACH_TO_COLOR, APPROACH_TO_DISPLAY_NAME

APPROACH_NAME_TO_KEY = {v: k for k, v in APPROACH_TO_DISPLAY_NAME.items()}


def _main(outfile: Path) -> None:
    plt.style.use(Path(__file__).parent / "custom.mplstyle")

    data = [
        {"Trial": 1, "Approach": "CBTL (Ours)", "Bite 1": 1, "Bite 2": 0, "Drink 1": 0},
        {"Trial": 1, "Approach": "No Learning", "Bite 1": 1, "Bite 2": 1, "Drink 1": 1},
        {"Trial": 2, "Approach": "CBTL (Ours)", "Bite 1": 1, "Bite 2": 0, "Drink 1": 0},
        {"Trial": 2, "Approach": "No Learning", "Bite 1": 1, "Bite 2": 1, "Drink 1": 1},
        {"Trial": 3, "Approach": "CBTL (Ours)", "Bite 1": 1, "Bite 2": 0, "Drink 1": 0},
        {"Trial": 3, "Approach": "No Learning", "Bite 1": 1, "Bite 2": 1, "Drink 1": 1},
        {"Trial": 4, "Approach": "CBTL (Ours)", "Bite 1": 1, "Bite 2": 0, "Drink 1": 0},
        {"Trial": 4, "Approach": "No Learning", "Bite 1": 1, "Bite 2": 1, "Drink 1": 1},
        {"Trial": 5, "Approach": "CBTL (Ours)", "Bite 1": 1, "Bite 2": 0, "Drink 1": 0},
        {"Trial": 5, "Approach": "No Learning", "Bite 1": 1, "Bite 2": 1, "Drink 1": 1},
    ]
    df = pd.DataFrame(data)

    # Map approaches to internal keys and display names
    df["ApproachKey"] = df["Approach"].map(APPROACH_NAME_TO_KEY)
    df["ApproachLabel"] = df["ApproachKey"].map(APPROACH_TO_DISPLAY_NAME)

    # Convert to long format
    df_long = df.melt(
        id_vars=["Trial", "ApproachKey", "ApproachLabel"],
        value_vars=["Bite 1", "Bite 2", "Drink 1"],
        var_name="Action",
        value_name="User Complaint",
    )

    # Average over trials
    df_avg = (
        df_long.groupby(["ApproachKey", "ApproachLabel", "Action"], as_index=False)
        .mean()
        .sort_values(by="ApproachLabel")
    )

    # Plot
    palette = [APPROACH_TO_COLOR[k] for k in df_avg["ApproachKey"].unique()]
    plt.figure(figsize=(6, 4))
    sns.lineplot(
        data=df_avg,
        x="Action",
        y="User Complaint",
        hue="ApproachLabel",
        palette=palette,
        marker="o",
    )
    plt.legend(title=None)
    ax = plt.gca()
    ax.set_ylim(-0.05, 1.05)
    ticks = ax.yaxis.get_minor_ticks()
    ticks[0].tick1line.set_visible(False)
    ticks[0].tick2line.set_visible(False)
    ticks[-1].tick1line.set_visible(False)
    ticks[-1].tick2line.set_visible(False)
    ticks = ax.xaxis.get_minor_ticks()
    for tick in ticks:
        tick.set_visible(False)
    plt.ylabel("User Complaints")
    plt.xlabel("")
    plt.title("Real Robot Results")
    plt.savefig(outfile, dpi=1000, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("outfile", type=Path)
    args = parser.parse_args()
    _main(args.outfile)
