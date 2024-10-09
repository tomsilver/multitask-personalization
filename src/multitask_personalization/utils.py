"""General utility functions."""

from functools import lru_cache
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Collection, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose3D
from skimage.transform import resize  # pylint: disable=no-name-in-module

from multitask_personalization.structs import HashableComparable, Image

_T = TypeVar("_T", bound=HashableComparable)


def topological_sort(l: Collection[_T], pairs: Collection[tuple[_T, _T]]) -> list[_T]:
    """Create an ordered verison of l that obeys pairwise > relations."""
    # Create the TopologicalSorter object.
    ts = TopologicalSorter({x: {} for x in l})
    for x, y in pairs:
        ts.add(x, y)
    try:
        sorted_list = list(ts.static_order())
    except CycleError:
        raise ValueError("The given constraints form a cycle and cannot be satisfied")

    return sorted_list


def load_avatar_asset(filename: str) -> Image:
    """Load an image of an avatar."""
    asset_dir = Path(__file__).parent / "assets" / "avatars"
    image_file = asset_dir / filename
    return plt.imread(image_file)


@lru_cache(maxsize=None)
def get_avatar_by_name(avatar_name: str, tilesize: int) -> Image:
    """Helper for rendering."""
    if avatar_name == "robot":
        im = load_avatar_asset("robot.png")
    elif avatar_name == "bunny":
        im = load_avatar_asset("bunny.png")
    elif avatar_name == "obstacle":
        im = load_avatar_asset("obstacle.png")
    elif avatar_name == "fire":
        im = load_avatar_asset("fire.png")
    elif avatar_name == "hidden":
        im = load_avatar_asset("hidden.png")
    else:
        raise ValueError(f"No asset for {avatar_name} known")
    shape = (tilesize, tilesize, 3)
    return resize(im[:, :, :3], shape, preserve_range=True)  # type: ignore


def render_avatar_grid(avatar_grid: NDArray, tilesize: int = 64) -> Image:
    """Helper for rendering."""
    height, width = avatar_grid.shape
    canvas = np.zeros((height * tilesize, width * tilesize, 3))

    for r in range(height):
        for c in range(width):
            avatar_name: str | None = avatar_grid[r, c]
            if avatar_name is None:
                continue
            im = get_avatar_by_name(avatar_name, tilesize)
            canvas[
                r * tilesize : (r + 1) * tilesize,
                c * tilesize : (c + 1) * tilesize,
            ] = im

    return (255 * canvas).astype(np.uint8)


def fig2data(fig: plt.Figure) -> Image:
    """Convert matplotlib figure into Image."""
    fig.canvas.draw()
    assert isinstance(fig.canvas, FigureCanvasAgg)
    return np.array(fig.canvas.renderer.buffer_rgba())


def sample_spherical(center: Pose3D, radius: float, rng: np.random.Generator) -> Pose3D:
    """Based on https://stackoverflow.com/questions/33976911/"""
    # Sample on the unit sphere.
    vec = rng.normal(size=(3,))
    vec /= np.linalg.norm(vec, axis=0)
    # Scale.
    vec = radius * vec
    # Translate.
    vec = np.add(center, vec)
    return vec.tolist()
