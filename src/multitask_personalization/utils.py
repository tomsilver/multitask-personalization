"""General utility functions."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose3D
from scipy.spatial.transform import Rotation as R

from multitask_personalization.structs import CSP, CSPSampler, CSPVariable

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


def solve_csp(
    csp: CSP,
    initialization: dict[CSPVariable, Any],
    samplers: list[CSPSampler],
    rng: np.random.Generator,
) -> dict[CSPVariable, Any]:
    """A very naive solver for CSPs."""
    sol = initialization
    while True:
        if csp.check_solution(sol):
            return sol
        sampler = samplers[rng.choice(len(samplers))]
        sol = sampler.sample(sol, rng)
