"""General utility functions."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose3D
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

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


def solve_csp(
    csp: CSP,
    initialization: dict[CSPVariable, Any],
    samplers: list[CSPSampler],
    rng: np.random.Generator,
    max_iters: int = 100_000,
    min_num_satisfying_solutions: int = 50,
    show_progress_bar: bool = True,
) -> dict[CSPVariable, Any] | None:
    """A very naive solver for CSPs."""
    sol = initialization.copy()
    best_satisfying_sol: dict[CSPVariable, Any] | None = None
    best_satisfying_cost: float = np.inf
    num_satisfying_solutions = 0
    for _ in (pbar := tqdm(range(max_iters), disable=not show_progress_bar)):
        pbar.set_description(f"Found {num_satisfying_solutions} solns")
        if csp.check_solution(sol):
            num_satisfying_solutions += 1
            if csp.cost is None:
                return sol
            cost = csp.get_cost(sol)
            if cost < best_satisfying_cost:
                best_satisfying_cost = cost
                best_satisfying_sol = sol
            if num_satisfying_solutions >= min_num_satisfying_solutions:
                return best_satisfying_sol
        sampler = samplers[rng.choice(len(samplers))]
        partial_sol = sampler.sample(sol, rng)
        sol = sol.copy()
        sol.update(partial_sol)
    return best_satisfying_sol
