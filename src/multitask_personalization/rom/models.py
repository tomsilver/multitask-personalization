"""ROM models."""

import abc
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
import torch
from numpy.typing import NDArray
from scipy.spatial import KDTree

from multitask_personalization.envs.pybullet.pybullet_human_spec import (
    HumanSpec,
    create_human_from_spec,
)
from multitask_personalization.rom.implicit_mlp import MLPROMClassifierTorch
from multitask_personalization.utils import (
    DIMENSION_LIMITS,
    DIMENSION_NAMES,
    denormalize_samples,
    rotation_matrix_x,
    rotation_matrix_y,
    rotmat2euler,
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
        self._human = create_human_from_spec(
            human_spec, self._rng, self._physics_client_id
        )

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

    def _run_human_fk(self, joint_positions: NDArray) -> NDArray:
        """Run forward kinematics for the human given joint positions."""

        # Transform from collected data angle space into pybullet angle space.
        shoulder_aa, shoulder_fe, shoulder_rot, elbow_flexion = joint_positions
        shoulder_aa = -shoulder_aa
        shoulder_rot -= 90
        shoulder_rot = -shoulder_rot
        local_rot_mat = (
            rotation_matrix_y(90)
            @ rotation_matrix_x(shoulder_aa)
            @ rotation_matrix_y(shoulder_fe)
            @ rotation_matrix_x(shoulder_rot)
        )
        transformed_angles = rotmat2euler(local_rot_mat, seq="YZX")
        shoulder_x = transformed_angles[0] - 90
        shoulder_y = transformed_angles[1]
        shoulder_z = 180 - transformed_angles[2]
        elbow = elbow_flexion

        current_right_arm_joint_angles = self._human.get_joint_angles(
            self._human.right_arm_joints
        )
        target_right_arm_angles = np.copy(current_right_arm_joint_angles)
        shoulder_x_index = self._human.j_right_shoulder_x
        shoulder_y_index = self._human.j_right_shoulder_y
        shoulder_z_index = self._human.j_right_shoulder_z
        elbow_index = self._human.j_right_elbow

        target_right_arm_angles[shoulder_x_index] = np.radians(shoulder_x)
        target_right_arm_angles[shoulder_y_index] = np.radians(shoulder_y)
        target_right_arm_angles[shoulder_z_index] = np.radians(shoulder_z)
        target_right_arm_angles[elbow_index] = np.radians(elbow)
        other_idxs = set(self._human.right_arm_joints) - {
            shoulder_x_index,
            shoulder_y_index,
            shoulder_z_index,
            elbow_index,
        }
        assert np.allclose([target_right_arm_angles[i] for i in other_idxs], 0.0)
        self._human.set_joint_angles(
            self._human.right_arm_joints, target_right_arm_angles, use_limits=False
        )
        right_wrist_pos, _ = self._human.get_pos_orient(self._human.right_wrist)
        return right_wrist_pos

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


class GroundTruthROMModel(ROMModel):
    """ROM model constructed from ground truth data."""

    def __init__(
        self,
        human_spec: HumanSpec,
        ik_distance_threshold: float = 1e-1,
        seed: int = 0,
    ) -> None:
        super().__init__(human_spec, seed=seed)
        self._subject = human_spec.subject_id
        self._condition = human_spec.condition
        self._ik_distance_threshold = ik_distance_threshold
        # Load array of points from rom/data/rom_points.pkl.
        with open(
            f"src/multitask_personalization/rom/data/{self._subject}_"
            + f"{self._condition}_dense_points.pkl",
            "rb",
        ) as f:
            self._reachable_joints = pickle.load(f)
        logging.info(
            f"Loaded {len(self._reachable_joints)} points"
            + f" from {self._subject}_{self._condition}_dense_points.pkl"
        )
        # Load reachable points from cache if available, otherwise generate and cache
        cache_dir = Path(__file__).parent / "cache"
        # use pathlib Path for path
        reachable_points_cache_path = (
            cache_dir / f"{human_spec.subject_id}_{human_spec.condition}_"
            f"{human_spec.gender}_{human_spec.impairment}_reachable_points.pkl"
        )
        if reachable_points_cache_path.exists():
            with open(reachable_points_cache_path, "rb") as f:
                self._reachable_points = pickle.load(f)
            logging.info("Loaded reachable points from cache.")
        else:
            # Create reachable point cloud using human FK.
            self._reachable_points = [
                self._run_human_fk(point) for point in self._reachable_joints
            ]
            cache_dir.mkdir(exist_ok=True)
            # Cache reachable points
            with open(
                reachable_points_cache_path,
                "wb",
            ) as f:
                pickle.dump(self._reachable_points, f)
            logging.info("Cached reachable points.")
        self._reachable_kd_tree = KDTree(self._reachable_points)

        # Uncomment for debugging.
        # self._visualize_reachable_points()

    def save(self, model_dir: Path) -> None:
        pass

    def load(self, model_dir: Path) -> None:
        pass

    def check_position_reachable(
        self,
        position: NDArray,
    ) -> bool:
        distance, _ = self._reachable_kd_tree.query(position)
        return distance < self._ik_distance_threshold

    def get_position_reachable_logprob(
        self,
        position: NDArray,
    ) -> float:
        raise NotImplementedError

    def sample_reachable_position(self, rng: np.random.Generator) -> NDArray:
        return rng.choice(self._reachable_points)


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
    ) -> None:
        super().__init__(human_spec, seed=seed)
        self._min_possible_radius = min_possible_radius
        self._max_possible_radius = max_possible_radius
        origin, _ = self._human.get_pos_orient(self._human.right_wrist)
        self._sphere_center = origin

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


class LearnedROMModel(TrainableROMModel):
    """ROM model learned from data."""

    def __init__(
        self,
        human_spec: HumanSpec,
        ik_distance_threshold: float = 1e-1,
    ) -> None:
        super().__init__(human_spec)
        self._ik_distance_threshold = ik_distance_threshold
        self._rom_model = MLPROMClassifierTorch(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self._rom_model.load()
        self._rom_model_context_parameters = np.array([0.0251, -0.2047, 0.3738, 0.1586])
        """Parameters generated from functional score encoder as samples from.

        the learned distribution
        - [-0.1040, -0.2353, 0.2436, 0.0986]
        - [-0.1230, -0.1877,  0.2162,  0.1826]
        - [-0.0645, -0.2141,  0.2723,  0.0986]
        """

        # generate a dense grid of joint-space points
        grids = []
        for dim_name in DIMENSION_NAMES:
            grids.append(
                np.linspace(
                    DIMENSION_LIMITS[dim_name][0],
                    DIMENSION_LIMITS[dim_name][1],
                    40,
                )
            )
        grid = np.meshgrid(*grids)
        joint_angle_samples = np.vstack([g.ravel() for g in grid]).T
        # normalize samples using DIMENSION_LIMITS
        joint_angle_samples = np.array(joint_angle_samples)
        for i, dim_name in enumerate(DIMENSION_NAMES):
            dim_min, dim_max = DIMENSION_LIMITS[dim_name]
            joint_angle_samples[:, i] = (joint_angle_samples[:, i] - dim_min) / (
                dim_max - dim_min
            )
        self._dense_joint_samples = joint_angle_samples
        self.set_trainable_parameters(self._rom_model_context_parameters)
        logging.info("Learned ROM model created")

        # Uncomment for debugging.
        # self._visualize_reachable_points()

    def save(self, model_dir: Path) -> None:
        raise NotImplementedError

    def load(self, model_dir: Path) -> None:
        raise NotImplementedError

    def check_position_reachable(
        self,
        position: NDArray,
    ) -> bool:
        distance, _ = self._reachable_kd_tree.query(position)
        return distance < self._ik_distance_threshold

    def sample_reachable_position(self, rng: np.random.Generator) -> NDArray:
        return rng.choice(self._reachable_points)

    def get_position_reachable_logprob(
        self,
        position: NDArray,
    ) -> float:
        raise NotImplementedError

    def get_trainable_parameters(self) -> Any:
        return self._rom_model_context_parameters.copy()

    def set_trainable_parameters(self, params: Any) -> None:
        assert isinstance(params, np.ndarray)
        self._rom_model_context_parameters = params.copy()
        # forward pass through the model to get the dense grid of reachable
        # points in task space
        context_parameters = np.tile(
            self._rom_model_context_parameters, (len(self._dense_joint_samples), 1)
        )
        input_data = np.concatenate(
            (context_parameters, self._dense_joint_samples), axis=1
        )
        preds = (
            self._rom_model.classify(
                torch.tensor(input_data, dtype=torch.float32).to(self._rom_model.device)
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.bool_)
        )
        num_pos = np.sum(preds == 1)
        num_neg = np.sum(preds == 0)
        logging.info(f"Number of positive samples: {num_pos}")
        logging.info(f"Number of negative samples: {num_neg}")

        self._reachable_joints = denormalize_samples(
            self._dense_joint_samples[preds == 1]
        )
        self._reachable_points = [
            self._run_human_fk(point) for point in self._reachable_joints
        ]
        assert len(self._reachable_points) != 0, "No reachable points."
        self._reachable_kd_tree = KDTree(self._reachable_points)
        logging.info(
            f"Updated ROM model parameters, resulting in {len(self._reachable_points)}"
            " reachable points."
        )

    def train(self, data: list[tuple[NDArray, bool]]) -> None:
        raise NotImplementedError("Figure this out in a future PR...")
