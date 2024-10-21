"""ROM models."""

import abc
import pickle

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial import KDTree

from multitask_personalization.rom.implicit_mlp import MLPROMClassifierTorch
from multitask_personalization.utils import (
    DIMENSION_LIMITS,
    DIMENSION_NAMES,
    denormalize_samples,
)


class ROMModel(abc.ABC):
    """Base class for ROM models."""

    def __init__(self) -> None:
        self._reachable_points: list[NDArray] = []
        self._reachable_kd_tree: KDTree = KDTree(np.array([[0, 0]]))
        self._upd_reachable: bool = True

    @abc.abstractmethod
    def get_reachable_joints(self) -> NDArray:
        """Get the reachable joints."""

    @abc.abstractmethod
    def set_reachable_points(self, reachable_points: list[NDArray]) -> None:
        """Set the reachable points."""

    @abc.abstractmethod
    def get_reachable_points(self) -> list[NDArray]:
        """Get the reachable points."""

    @abc.abstractmethod
    def check_position_reachable(self, position: NDArray) -> bool:
        """Check if a position is reachable."""

    @abc.abstractmethod
    def sample_reachable_position(self) -> NDArray:
        """Sample a reachable position."""


class GroundTruthROMModel(ROMModel):
    """ROM model constructed from ground truth data."""

    def __init__(
        self,
        rng: np.random.Generator,
        subject: int,
        condition: str,
        ik_distance_threshold: float = 1e-1,
    ) -> None:
        super().__init__()
        self._rng = rng
        self._subject = subject
        self._condition = condition
        self._ik_distance_threshold = ik_distance_threshold
        # load array of points from rom/data/rom_points.pkl as self._reachable_joints
        with open(
            f"src/multitask_personalization/rom/data/{subject}_{condition}_"
            + "dense_points.pkl",
            "rb",
        ) as f:
            self._reachable_joints = pickle.load(f)
        print(
            f"Loaded {len(self._reachable_joints)} points"
            + " from {subject}_{condition}_dense_points.pkl"
        )

    def get_reachable_joints(self) -> NDArray:
        return self._reachable_joints

    def set_reachable_points(self, reachable_points: list[NDArray]) -> None:
        self._reachable_points = reachable_points
        assert len(self._reachable_points) != 0, "No reachable points."
        self._reachable_kd_tree = KDTree(self._reachable_points)
        self._upd_reachable = False

    def get_reachable_points(self) -> list[NDArray]:
        return self._reachable_points

    def check_position_reachable(self, position: NDArray) -> bool:
        assert not self._upd_reachable, "Must set reachable points first."
        distance, _ = self._reachable_kd_tree.query(position)
        return distance < self._ik_distance_threshold

    def sample_reachable_position(self) -> NDArray:
        assert not self._upd_reachable, "Must set reachable points first."
        return self._rng.choice(self._reachable_points)


class LearnedROMModel(ROMModel):
    """ROM model learned from data."""

    def __init__(
        self, rng: np.random.Generator, ik_distance_threshold: float = 1e-1
    ) -> None:
        super().__init__()
        self._rng = rng
        self._ik_distance_threshold = ik_distance_threshold
        self._rom_model = MLPROMClassifierTorch(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self._rom_model.load()
        self._parameter_size = 4
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
        self.update_parameters(self._rom_model_context_parameters)
        print("Learned ROM model created")

    def get_parameter_size(self) -> int:
        """Get the size of the parameter."""
        return self._parameter_size

    def get_rom_model_context_parameters(self) -> NDArray:
        """Get the ROM model context parameters."""
        return self._rom_model_context_parameters

    def update_parameters(self, parameters: NDArray) -> None:
        """Update the ROM model parameters."""
        self._rom_model_context_parameters = parameters
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
        print(f"Number of positive samples: {num_pos}")
        print(f"Number of negative samples: {num_neg}")

        self._reachable_joints = denormalize_samples(
            self._dense_joint_samples[preds == 1]
        )
        self._reachable_points = []
        self._reachable_kd_tree = None
        self._upd_reachable = True
        print(
            "Updated ROM model parameters and reachable joints."
            + "Need to update reachable points."
        )

    def get_reachable_joints(self) -> NDArray:
        return self._reachable_joints

    def set_reachable_points(self, reachable_points: list[NDArray]) -> None:
        self._reachable_points = reachable_points
        assert len(self._reachable_points) != 0, "No reachable points."
        self._reachable_kd_tree = KDTree(self._reachable_points)
        self._upd_reachable = False

    def get_reachable_points(self) -> list[NDArray]:
        return self._reachable_points

    def check_position_reachable(self, position: NDArray) -> bool:
        assert not self._upd_reachable, "Must set reachable points first."
        distance, _ = self._reachable_kd_tree.query(position)
        return distance < self._ik_distance_threshold

    def sample_reachable_position(self) -> NDArray:
        assert not self._upd_reachable, "Must set reachable points first."
        return self._rng.choice(self._reachable_points)
