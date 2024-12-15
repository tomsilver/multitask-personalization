"""ROM models."""

import abc
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, multiply_poses
from scipy.spatial import KDTree

from multitask_personalization.envs.pybullet.pybullet_human import (
    HumanSpec,
    create_human_from_spec,
)
from multitask_personalization.utils import (
    sample_within_sphere,
)


class ROMModel(abc.ABC):
    """Base class for ROM models."""

    def __init__(
        self,
        human_spec: HumanSpec,
        seed: int = 0,
    ) -> None:
        # Create human.
        # Uncomment for debugging
        # from pybullet_helpers.gui import create_gui_connection
        # self._physics_client_id = create_gui_connection()
        self._physics_client_id = p.connect(p.DIRECT)
        self._rng = np.random.default_rng(seed)
        self._human = create_human_from_spec(human_spec, self._physics_client_id)

        self._reachable_points: list[NDArray] = []
        self._reachable_kd_tree: KDTree = KDTree(np.array([[0, 0]]))

    @abc.abstractmethod
    def save(self, model_dir: Path) -> None:
        """Save sufficient information about the model."""

    @abc.abstractmethod
    def load(self, model_dir: Path) -> None:
        """Load from a saved model dir."""

    @abc.abstractmethod
    def check_position_reachable(
        self,
        position: NDArray,
    ) -> bool:
        """Check if a position is reachable."""

    @abc.abstractmethod
    def sample_reachable_position(self, rng: np.random.Generator) -> NDArray:
        """Sample a reachable position."""

    @abc.abstractmethod
    def get_position_reachable_logprob(
        self,
        position: NDArray,
    ) -> float:
        """Get the log probability that the position is reachable."""

    def _visualize_reachable_points(
        self,
        n: int = 300,
        color: tuple[float, float, float, float] = (0.5, 1.0, 0.2, 0.6),
    ) -> None:
        # Randomly sample n reachable points.
        sampled_points = np.array(
            [
                self._reachable_points[i]
                for i in self._rng.choice(len(self._reachable_points), n, replace=False)
            ]
        )
        # Create a visual shape for each sampled point.
        for _, point in enumerate(sampled_points):
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.04,
                rgbaColor=color,
                physicsClientId=self._physics_client_id,
            )

            p.createMultiBody(
                baseVisualShapeIndex=visual_shape_id,
                basePosition=point,
                physicsClientId=self._physics_client_id,
            )

        while True:
            p.stepSimulation(self._physics_client_id)


class TrainableROMModel(ROMModel):
    """Base class for trainable ROM models."""

    @abc.abstractmethod
    def get_trainable_parameters(self) -> Any:
        """Access the current trainable parameter values."""

    @abc.abstractmethod
    def set_trainable_parameters(self, params: Any) -> None:
        """Set the trainable parameter values."""

    @abc.abstractmethod
    def train(self, data: list[tuple[NDArray, bool]]) -> None:
        """Update trainable parameters given a dataset of (position, label)."""

    def get_metrics(self) -> dict[str, float]:
        """Optionally report metrics, e.g., learned parameters."""
        return {}


class SphericalROMModel(TrainableROMModel):
    """ROM model with spherical reachability."""

    def __init__(
        self,
        human_spec: HumanSpec,
        seed: int = 0,
        min_possible_radius: float = 0.25,
        max_possible_radius: float = 1.25,
        origin_distance: float = 0.2,
    ) -> None:
        super().__init__(human_spec, seed=seed)
        self._min_possible_radius = min_possible_radius
        self._max_possible_radius = max_possible_radius
        # Set the origin to be in front of the hand.
        ee_pose = self._human.get_end_effector_pose()
        origin_tf = Pose((0.0, 0.0, origin_distance))
        origin_pose = multiply_poses(ee_pose, origin_tf)
        self._sphere_center = origin_pose.position

        # Uncomment for debugging.
        # self._reachable_points = self._sample_spherical_points(n=500)
        # self._visualize_reachable_points()

    def save(self, model_dir: Path) -> None:
        outfile = model_dir / "spherical_rom_params.json"
        params = {
            "min_possible_radius": self._min_possible_radius,
            "max_possible_radius": self._max_possible_radius,
        }
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(params, f)

    def load(self, model_dir: Path) -> None:
        outfile = model_dir / "spherical_rom_params.json"
        with open(outfile, "r", encoding="utf-8") as f:
            params = json.load(f)
        self._min_possible_radius = params["min_possible_radius"]
        self._max_possible_radius = params["max_possible_radius"]

    @property
    def _radius(self) -> float:
        return (self._max_possible_radius + self._min_possible_radius) / 2

    def _distance_to_center(
        self,
        position: NDArray,
    ) -> float:
        return float(np.linalg.norm(np.subtract(position, self._sphere_center)))

    def get_trainable_parameters(self) -> Any:
        return (self._min_possible_radius, self._max_possible_radius)

    def set_trainable_parameters(self, params: Any) -> None:
        min_radius, max_radius = params
        self._min_possible_radius = min_radius
        self._max_possible_radius = max_radius

    def check_position_reachable(
        self,
        position: NDArray,
    ) -> bool:
        return self._distance_to_center(position) < self._radius

    def sample_reachable_position(self, rng: np.random.Generator) -> NDArray:
        return np.array(sample_within_sphere(self._sphere_center, self._radius, rng))

    def get_position_reachable_logprob(
        self,
        position: NDArray,
    ) -> float:
        distance = self._distance_to_center(position)
        if distance <= self._min_possible_radius:
            return 0.0  # definitely reachable
        if distance >= self._max_possible_radius:
            return -np.inf  # definitely not reachable
        return np.log(0.5)  # uncertain

    def _sample_spherical_points(self, n: int = 500) -> list[NDArray]:
        return [
            np.array(sample_within_sphere(self._sphere_center, self._radius, self._rng))
            for _ in range(n)
        ]

    def train(self, data: list[tuple[NDArray, bool]]) -> None:
        # Find decision boundary between maximal positive and minimal negative.
        logging.info(f"Training SphericalROMModel with {len(data)} data")
        for position, label in data:
            dist = self._distance_to_center(position)
            if label:
                # We've found a positive data point that is farther than what
                # we've previously seen, so increase the min possible radius.
                self._min_possible_radius = max(dist, self._min_possible_radius)
            else:
                # We've found a negative data point that is closer than what
                # we've previously seen, so decrease the max possible radius.
                self._max_possible_radius = min(dist, self._max_possible_radius)
        logging.info(
            f"Updating SphericalROMModel: min={self._min_possible_radius}, "
            f"max={self._max_possible_radius}"
        )

    def get_metrics(self) -> dict[str, float]:
        return {
            "spherical_rom_min_radius": self._min_possible_radius,
            "spherical_rom_max_radius": self._max_possible_radius,
        }
