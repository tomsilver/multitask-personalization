"""Create book example results plots."""

import argparse
from pathlib import Path

import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

BOOK_TO_COLOR = {
    "1984": "#1b9e77",
    "Fahrenheit 451": "#d95f02",
    "Cosmos": "#7570b3",
    "Double Helix": "#e7298a",
}

PROBABILITIES = {
    0: {
        "1984": 0.40371595937943944,
        "Fahrenheit 451": 0.4017815357783198,
        "Cosmos": 0.40084298247383565,
        "Double Helix": 0.4006559924582323,
    },
    148: {
        "1984": 0.998850671372682,
        "Fahrenheit 451": 0.816852438461953,
        "Cosmos": 0.39317202766095644,
        "Double Helix": 0.3906668947996979,
    },
    494: {
        "1984": 0.9880735166242507,
        "Fahrenheit 451": 0.836520303973082,
        "Cosmos": 0.11389223764599972,
        "Double Helix": 0.00017839858942570653,
    },
    782: {
        "1984": 0.9981945341403206,
        "Fahrenheit 451": 0.9988988446238175,
        "Cosmos": 0.099809881381634,
        "Double Helix": 5.6515823041317815e-05,
    },
    1138: {
        "1984": 0.9991384789772672,
        "Fahrenheit 451": 0.9998479321254011,
        "Cosmos": 9.831035356917004e-06,
        "Double Helix": 5.558122367749702e-06,        
    }
}



def _main(outfile: Path) -> None:
    plt.style.use(Path(__file__).parent / "custom.mplstyle")

    rows = []
    for timestep, book_probs in PROBABILITIES.items():
        for book, prob in book_probs.items():
            rows.append({"timestep": 100 * timestep / max(PROBABILITIES), "book": book, "probability": prob})
    df = pd.DataFrame(rows)

    # Plot
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="timestep",
        y="probability",
        hue="book",
        palette=BOOK_TO_COLOR,
        marker="o"
    )
    plt.legend(title=None, fontsize=28, loc="center right")

    plt.xlabel("")
    plt.ylabel("")
    plt.xlim((0, 105))
    plt.ylim((-0.05, 1.05))
    plt.tight_layout()
    plt.savefig(outfile, dpi=800)
    print(f"Plot saved to {outfile}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("outfile", type=Path)
    args = parser.parse_args()
    _main(args.outfile)
