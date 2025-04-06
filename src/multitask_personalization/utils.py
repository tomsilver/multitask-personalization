"""General utility functions."""

import os
from pathlib import Path
from typing import Any

import graphviz
import numpy as np
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose3D
from scipy.spatial.transform import Rotation as R

from multitask_personalization.structs import CSP, CSPVariable

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


def euler2rotmat(euler: NDArray, seq: str = "ZXY") -> NDArray:
    """Inverse of rotmat2euler."""
    rotation = R.from_euler(seq, euler, degrees=True)
    return rotation.as_matrix()


def sample_within_sphere(
    center: Pose3D, min_radius: float, max_radius: float, rng: np.random.Generator
) -> Pose3D:
    """Sample a random point within a sphere of given radius and center."""
    return sample_on_sphere(center, rng.uniform(min_radius, max_radius), rng)


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


class Bounded1DClassifier:
    """Predicts the probability that 1D x is true or false given data, and
    given that there is some interval with known lower and upper bounds that
    determines the classifier.

    See notebooks/ for more explanation.
    """

    def __init__(self, a_lo: float, b_hi: float) -> None:
        self.a_lo = a_lo
        self.b_hi = b_hi

        self.x1 = a_lo
        self.x2 = a_lo
        self.x3 = b_hi
        self.x4 = b_hi

        self.incremental_X: list[float] = []
        self.incremental_Y: list[bool] = []

    def _fit(self, X: list[float], Y: list[bool]) -> None:
        """Fit the model parameters."""
        X_pos, X_neg = set(), set()
        for x, y in zip(X, Y, strict=True):
            if y:
                X_pos.add(x)
            else:
                X_neg.add(x)
        # Can't fit if there is no positive data.
        if not X_pos:
            return
        m = next(iter(X_pos))
        X_neg_lo, X_neg_hi = set(), set()
        for x in X_neg:
            if x < m:
                X_neg_lo.add(x)
            else:
                X_neg_hi.add(x)
        self.x1 = max(X_neg_lo | {self.a_lo})
        self.x2 = min(X_pos)
        self.x3 = max(X_pos)
        self.x4 = min(X_neg_hi | {self.b_hi})

    def fit(self, X: list[float], Y: list[bool]) -> None:
        """Discard any previous data and fit to the new data."""
        self.incremental_X = list(X)
        self.incremental_Y = list(Y)
        self._fit(self.incremental_X, self.incremental_Y)

    def fit_incremental(self, X: list[float], Y: list[bool]) -> None:
        """Accumulate training data and re-fit."""
        self.incremental_X.extend(X)
        self.incremental_Y.extend(Y)
        self._fit(self.incremental_X, self.incremental_Y)

    def predict_proba(self, X: list[float]) -> list[float]:
        """Batch predict class probabilities."""
        X_arr = np.array(X)
        return np.piecewise(
            X_arr,
            [
                X_arr < self.x1,
                (self.x1 <= X_arr) & (X_arr < self.x2),
                (self.x2 <= X_arr) & (X_arr < self.x3),
                (self.x3 <= X_arr) & (X_arr < self.x4),
                X_arr >= self.x4,
            ],
            [
                lambda _: 0,
                lambda x: (x - self.x1) / (self.x2 - self.x1),
                lambda _: 1,
                lambda x: 1 - (x - self.x3) / (self.x4 - self.x3),
                lambda _: 0,
            ],
        ).tolist()

    def get_save_state(self) -> dict[str, Any]:
        """Get everything needed to restore the model later."""
        return {
            "x1": self.x1,
            "x2": self.x2,
            "x3": self.x3,
            "x4": self.x4,
            "incremental_X": self.incremental_X,
            "incremental_Y": self.incremental_Y,
        }

    def load_from_state(self, state_dict: dict[str, Any]) -> None:
        """Load a model from a dictionary returned by get_save_state()."""
        self.x1 = state_dict["x1"]
        self.x2 = state_dict["x2"]
        self.x3 = state_dict["x3"]
        self.x4 = state_dict["x4"]
        self.incremental_X = state_dict["incremental_X"]
        self.incremental_Y = state_dict["incremental_Y"]

    def get_summary(self) -> str:
        """Get a short human-readable summary of the current model."""
        return f"x1={self.x1:.3f} x2={self.x2:.3f} x3={self.x3:.3f} x4={self.x4:.4f}"


class Threshold1DModel:
    """A Bayesian 1D threshold model with a uniform prior on [min_theta,
    max_theta].

    The likelihood is a step function:
        P(y=1 | x, theta) = 1 if x >= theta, else 0
    so each observation constrains the feasible region for theta.
    The posterior is uniform over the intersection of those constraints.
    The posterior predictive for a new x integrates over that region.
    """

    def __init__(self, min_theta: float, max_theta: float):
        self.min_theta = min_theta
        self.max_theta = max_theta

        # We'll keep track of the "posterior" as an interval [post_min, post_max].
        # Initially, it is the entire prior range.
        self.post_min = min_theta
        self.post_max = max_theta

        self.incremental_X: list[float] = []
        self.incremental_Y: list[bool] = []

    def _update_posterior_from_data(self, X: list[float], Y: list[bool]) -> None:
        """Update the posterior interval [post_min, post_max] given the
        constraints from the data (X, Y)."""
        # Start from the prior each time we do a "complete" fit:
        self.post_min = self.min_theta
        self.post_max = self.max_theta

        for x_i, y_i in zip(X, Y):
            if y_i:
                # y_i = 1 => x_i >= theta => theta <= x_i
                # so post_max is at most x_i
                post_max = min(post_max, x_i)
            else:
                # y_i = 0 => x_i < theta => theta > x_i
                # so post_min is at least x_i
                post_min = max(post_min, x_i)
            # If the posterior interval becomes invalid, break early
            if self.post_min > self.post_max:
                break

    def fit(self, X: list[float], Y: list[bool]) -> None:
        """Discard previous data, then update the posterior to account for the
        new data."""
        self.incremental_X = list(X)
        self.incremental_Y = list(Y)
        self._update_posterior_from_data(self.incremental_X, self.incremental_Y)

    def fit_incremental(self, X: list[float], Y: list[bool]) -> None:
        """Append new data and update the posterior accordingly."""
        self.incremental_X.extend(X)
        self.incremental_Y.extend(Y)
        self._update_posterior_from_data(self.incremental_X, self.incremental_Y)

    def predict_proba(self, X: list[float]) -> list[float]:
        """Return the posterior predictive P(y=1 | x).

        This is the proportion of the posterior interval [post_min,
        post_max] over which (x >= theta).
        """
        # If the posterior is degenerate or invalid, we can handle that gracefully:
        length = self.post_max - self.post_min
        if length <= 0:
            # Posterior measure is zero => data is inconsistent =>
            # for demonstration, return 0.5 or something constant
            return [0.5] * len(X)

        probs = []
        for x_i in X:
            if x_i < self.post_min:
                p = 0.0
            elif x_i > self.post_max:
                p = 1.0
            else:
                # x_i in [post_min, post_max]
                # fraction of that interval that is <= x_i
                # i.e. measure([post_min, min(post_max, x_i)]) /
                #   measure([post_min, post_max])
                p = (x_i - self.post_min) / length
            probs.append(p)

        return probs

    def get_save_state(self) -> dict[str, Any]:
        """Get everything needed to restore the model later."""
        return {
            "post_min": self.post_min,
            "post_max": self.post_max,
            "incremental_X": self.incremental_X,
            "incremental_Y": self.incremental_Y,
        }

    def load_from_state(self, state_dict: dict[str, Any]) -> None:
        """Load a model from a dictionary returned by get_save_state()."""
        self.post_min = state_dict["post_min"]
        self.post_max = state_dict["post_max"]
        self.incremental_X = state_dict["incremental_X"]
        self.incremental_Y = state_dict["incremental_Y"]

    def get_summary(self) -> str:
        """Get a short human-readable summary of the current model."""
        return f"post_min={self.post_min:.3f}, post_max={self.post_max:.3f}"


def print_csp_sol(sol: dict[CSPVariable, Any]) -> None:
    """Useful for debugging."""
    print("-" * 80)
    for v in sorted(sol, key=lambda v: v.name):
        print(f"{v.name}: {sol[v]}")
    print()


def visualize_csp_graph(csp: CSP, outfile: Path, dpi: int = 250) -> None:
    """Save an image of the structure of the CSP."""
    intermediate_dot_file = outfile.parent / outfile.stem
    assert not intermediate_dot_file.exists()
    dot = graphviz.Graph(format=outfile.suffix[1:])
    variable_nodes = {v.name for v in csp.variables}
    constraint_nodes = {c.name for c in csp.constraints}
    for node in variable_nodes:
        dot.node(node, shape="circle")
    for node in constraint_nodes:
        dot.node(node, shape="box")
    for constraint in csp.constraints:
        for variable in constraint.variables:
            dot.edge(constraint.name, variable.name)
    dot.attr(dpi=str(dpi))
    dot.render(outfile.stem, directory=outfile.parent)
    os.remove(intermediate_dot_file)
    print(f"Wrote out to {outfile}")


class _NoChange:
    def __repr__(self):
        return "<NoChange>"


_NO_CHANGE = _NoChange()
