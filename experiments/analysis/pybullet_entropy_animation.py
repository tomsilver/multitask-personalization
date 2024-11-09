"""Create an animated plot showing entropy over time in pybullet CSP."""

import argparse
from pathlib import Path

import pandas as pd
from matplotlib import animation
from matplotlib import pyplot as plt

from multitask_personalization.envs.pybullet.pybullet_task_spec import PyBulletTaskSpec


def _create_spherical_radius_animation(
    csv_file: Path, outfile: Path, fps: int = 30, ground_truth_radius: float = 0.3
) -> None:
    df = pd.read_csv(csv_file)
    df = df.sort_values(by="step").reset_index(drop=True)

    # Set up the figure and axis.
    fig, ax = plt.subplots()
    ax.set_xlabel("Step")
    ax.set_ylabel("Radius")
    ax.set_title("Learned Spherical ROM Radius")
    x_min = -1
    x_max = max(df["step"]) + 1
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((-0.1, max(df["spherical_rom_max_radius"]) + 0.1))
    ax.plot(
        [x_min, x_max],
        [ground_truth_radius, ground_truth_radius],
        color="red",
        linestyle="dashed",
        label="Ground Truth",
    )

    # Initialize an empty line plot.
    (min_line,) = ax.plot([], [], "b-")
    (max_line,) = ax.plot([], [], "b-")
    fill = ax.fill_between([], [], [], color="blue", alpha=0.3)

    # Define the initialization function.
    def init():
        return min_line, max_line, fill

    # Define the animation function.
    def animate(i):
        nonlocal fill
        # Take the data up to step i.
        x = df["step"][: i + 1]
        min_y = df["spherical_rom_min_radius"][: i + 1]
        max_y = df["spherical_rom_max_radius"][: i + 1]
        fill.remove()
        fill = ax.fill_between(x, min_y, max_y, color="blue", alpha=0.3)
        min_line.set_data(x, min_y)
        max_line.set_data(x, max_y)
        return min_line, max_line, fill

    # Create the animation.
    ani = animation.FuncAnimation(
        fig, animate, frames=len(df) + 1, init_func=init, blit=True
    )

    # Save the animation.
    ani.save(outfile, writer="ffmpeg", fps=fps)
    print(f"Wrote out to {outfile}")

    plt.close()


def _create_book_entropy_animation(
    csv_file: Path, outfile: Path, fps: int = 30
) -> None:
    df = pd.read_csv(csv_file)
    df = df.sort_values(by="step").reset_index(drop=True)
    prefix = "entropy-"
    keys = [k for k in df.keys() if k.startswith(prefix)]
    books = [k[len(prefix) :] for k in keys]
    task_spec = PyBulletTaskSpec()
    book_colors = task_spec.book_rgbas
    assert len(books) == len(book_colors)

    # Set up the figure and axis.
    fig, ax = plt.subplots()
    ax.set_xlabel("Step")
    ax.set_ylabel("Entropy")
    ax.set_title("Book Preference Entropy")
    x_min = -1
    x_max = max(df["step"]) + 1
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((-0.1, max(df["spherical_rom_max_radius"]) + 0.1))

    # Initialize an empty line plot.
    lines = []
    for book, color in zip(books, book_colors, strict=True):
        (line,) = ax.plot([], [], label=book, color=color)
        lines.append(line)
    plt.legend(fontsize=8)

    # Define the initialization function.
    def init():
        return lines

    # Define the animation function.
    def animate(i):
        # Take the data up to step i.
        x = df["step"][: i + 1]
        for key, line in zip(keys, lines, strict=True):
            y = df[key][: i + 1]
            line.set_data(x, y)
        return lines

    # Create the animation.
    ani = animation.FuncAnimation(
        fig, animate, frames=len(df) + 1, init_func=init, blit=True
    )

    # Save the animation.
    ani.save(outfile, writer="ffmpeg", fps=fps)
    print(f"Wrote out to {outfile}")

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("outfile", type=Path)
    args = parser.parse_args()
    _create_spherical_radius_animation(args.csv_file, args.outfile)
    _create_book_entropy_animation(args.csv_file, args.outfile)
