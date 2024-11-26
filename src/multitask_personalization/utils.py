"""General utility functions."""

import numpy as np
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose3D
from scipy.spatial.transform import Rotation as R
from typing import Any
from multitask_personalization.structs import CSPVariable

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


def sample_within_sphere(
    center: Pose3D, radius: float, rng: np.random.Generator
) -> Pose3D:
    """Sample a random point within a sphere of given radius and center."""
    return sample_on_sphere(center, rng.uniform(0, radius), rng)


def sample_on_sphere(center: Pose3D, radius: float, rng: np.random.Generator) -> Pose3D:
    """Sample a random point on a sphere of given radius and center."""
    # Sample a random point on the unit sphere.
    vec = rng.normal(size=(3,))
    vec /= np.linalg.norm(vec, axis=0)

    vec = radius * vec

    # Translate to the center.
    vec = np.add(center, vec)
    return vec.tolist()


def bernoulli_entropy(log_p_true: float) -> float:
    """Compute entropy of a bernoulli RV given log prob.

    The input is in natural log units but the output entropy is in base
    2.
    """
    if np.isclose(log_p_true, 0) or np.isneginf(log_p_true):
        return 0.0
    p_true = np.exp(log_p_true)
    p_false = 1 - p_true
    log_p_false = np.log1p(-p_true)
    entropy_nats = -p_true * log_p_true - p_false * log_p_false
    entropy = entropy_nats / np.log(2)  # convert to base 2 for convention
    return entropy


def print_csp_sol(sol: dict[CSPVariable, Any]) -> None:
    """Useful for debugging."""
    print("-" * 80)
    for v in sorted(sol, key=lambda v: v.name):
        print(f"{v.name}: {sol[v]}")
    print()
