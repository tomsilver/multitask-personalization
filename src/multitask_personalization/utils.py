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
from scipy.spatial.transform import Rotation as R
from skimage.transform import resize  # pylint: disable=no-name-in-module

from multitask_personalization.structs import HashableComparable, Image

_T = TypeVar("_T", bound=HashableComparable)

DIMENSION_NAMES = ("shoulder_aa", "shoulder_fe", "shoulder_rot", "elbow_flexion")
DIMENSION_LIMITS = {
    "shoulder_aa": (-150, 150),
    "shoulder_fe": (0, 180),
    "shoulder_rot": (-50, 300),
    "elbow_flexion": (-10, 180),
}


def denormalize_samples(samples: NDArray, points_scale_factor=1.0) -> NDArray:
    """Denormalize samples."""
    samples_copy = samples.copy()
    for i, dim_name in enumerate(DIMENSION_NAMES):
        dim_min, dim_max = DIMENSION_LIMITS[dim_name]
        samples_copy[:, i] = (
            samples_copy[:, i] / points_scale_factor * (dim_max - dim_min) + dim_min
        )
    return samples_copy


def rotation_matrix_x(angle: float) -> NDArray:
    """Return the rotation matrix for a rotation around the x-axis."""
    angle = np.radians(angle)
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )


def rotation_matrix_y(angle: float) -> NDArray:
    """Return the rotation matrix for a rotation around the y-axis."""
    angle = np.radians(angle)
    return np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )


def rotmat2euler(rot_matrix: NDArray, seq: str = "ZXY") -> NDArray:
    """Convert a rotation matrix to euler angles."""
    rotation = R.from_matrix(rot_matrix)
    return rotation.as_euler(seq, degrees=True)


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
