"""Create main results plots."""

import argparse
from pathlib import Path
from typing import Callable

import seaborn as sns
from analysis_utils import combine_results_csvs
from matplotlib import pyplot as plt
from omegaconf import DictConfig

ENV_TO_DISPLAY_NAME = {
    "tiny": "Tiny",
    # "pybullet": "Home Library",
}

APPROACH_TO_DISPLAY_NAME = {
    "ours": "CBTL (Ours)",
    "nothing_personal": "Nothing Personal",
    "exploit_only": "Exploit Only",
}

# https://colorbrewer2.org/#type=diverging&scheme=PuOr&n=8
APPROACH_TO_COLOR = {
    "ours": "#542788",
    "nothing_personal": "#b35806",
    "exploit_only": "#fee0b6",
}


def _create_config_fn(
    env_name: str, approach_name: str
) -> Callable[[DictConfig], bool]:

    def _fn(cfg: DictConfig) -> bool:
        return cfg.env_name == env_name and cfg.approach_name == approach_name

    return _fn


def _main(results_dir: Path, outfile: Path) -> None:
    plt.style.use(Path(__file__).parent / "custom.mplstyle")

    plt.title("Main Results")

    _, axes = plt.subplots(1, len(ENV_TO_DISPLAY_NAME), squeeze=False)
    for ax, (env_name, env_display_name) in zip(axes[0], ENV_TO_DISPLAY_NAME.items()):
        ax.set_title(env_display_name)
        ax.set_xlabel("Simulated Execution Time (s)")
        ax.set_ylabel("Simulated User Satisfaction")
        for approach_name, approach_display_name in APPROACH_TO_DISPLAY_NAME.items():
            color = APPROACH_TO_COLOR[approach_name]
            config_fn = _create_config_fn(env_name, approach_name)
            df = combine_results_csvs(results_dir, config_fn=config_fn)
            sns.lineplot(
                data=df,
                x="training_execution_time",
                y="eval_mean_user_satisfaction",
                estimator="mean",
                errorbar="sd",
                ax=ax,
                color=color,
                label=approach_display_name,
            )

    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=500)
    print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("outfile", type=Path)
    args = parser.parse_args()
    _main(args.results_dir, args.outfile)