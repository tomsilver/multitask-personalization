"""ROM models."""

import abc
import json
import logging
from pathlib import Path

import numpy as np
import pybullet as p
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, multiply_poses
from scipy.spatial import KDTree

from multitask_personalization.envs.pybullet.pybullet_human import (
    HumanSpec,
    create_human_from_spec,
)
from multitask_personalization.utils import Bounded1DClassifier, sample_within_sphere


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
    def sample_position(self, rng: np.random.Generator) -> NDArray:
        """Sample any position, not necessarily reachable."""

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
        min_possible_radius: float = 0.0,
        max_possible_radius: float = 1.25,
        origin_distance: float = 0.2,
    ) -> None:
        super().__init__(human_spec, seed=seed)
        self._min_possible_radius = min_possible_radius
        self._max_possible_radius = max_possible_radius

        # Set the sphere origin to be in front of the hand.
        ee_pose = self._human.get_end_effector_pose()
        origin_tf = Pose((0.0, 0.0, origin_distance))
        origin_pose = multiply_poses(ee_pose, origin_tf)
        self._sphere_center = origin_pose.position

        # Fit a model to the sphere radius.
        self._radius_model = Bounded1DClassifier(
            min_possible_radius, max_possible_radius
        )

        # Uncomment for debugging.
        # self._reachable_points = self._sample_spherical_points(n=500)
        # self._visualize_reachable_points()

    def save(self, model_dir: Path) -> None:
        outfile = model_dir / "spherical_rom_params.json"
        params = self._radius_model.get_save_state()
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(params, f)

    def load(self, model_dir: Path) -> None:
        outfile = model_dir / "spherical_rom_params.json"
        with open(outfile, "r", encoding="utf-8") as f:
            params = json.load(f)
        self._radius_model.load_from_state(params)

    def _distance_to_center(
        self,
        position: NDArray,
    ) -> float:
        return float(np.linalg.norm(np.subtract(position, self._sphere_center)))

    def check_position_reachable(
        self,
        position: NDArray,
    ) -> bool:
        distance = self._distance_to_center(position)
        return self._radius_model.predict_proba([distance])[0] >= 0.5

    def sample_reachable_position(self, rng: np.random.Generator) -> NDArray:
        min_radius = (self._radius_model.x1 + self._radius_model.x2) / 2
        max_radius = (self._radius_model.x3 + self._radius_model.x4) / 2
        return np.array(
            sample_within_sphere(self._sphere_center, min_radius, max_radius, rng)
        )

    def sample_position(self, rng: np.random.Generator) -> NDArray:
        return np.array(
            sample_within_sphere(
                self._sphere_center,
                self._min_possible_radius,
                self._max_possible_radius,
                rng,
            )
        )

    def get_position_reachable_logprob(
        self,
        position: NDArray,
    ) -> float:
        distance = self._distance_to_center(position)
        prob = self._radius_model.predict_proba([distance])[0]
        return np.log(prob)

    def _sample_spherical_points(self, n: int = 500) -> list[NDArray]:
        return [self.sample_reachable_position(self._rng) for _ in range(n)]

    def train(self, data: list[tuple[NDArray, bool]]) -> None:
        # Find decision boundary between maximal positive and minimal negative.
        logging.info(f"Training SphericalROMModel with {len(data)} data")
        X, Y = [], []
        for position, label in data:
            dist = self._distance_to_center(position)
            X.append(dist)
            Y.append(label)
        logging.info(f"Last x, y: {X[-1]}, {Y[-1]}")
        self._radius_model.fit(X, Y)
        logging.info(f"Updating SphericalROMModel: {self._radius_model.get_summary()}")
