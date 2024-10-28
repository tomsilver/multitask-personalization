"""ROM models."""

import abc
import os
import pickle

import numpy as np
import pybullet as p
from numpy.typing import NDArray
from scipy.spatial import KDTree

from multitask_personalization.envs.pybullet.pybullet_human_spec import (
    HumanSpec,
    create_human_from_spec,
)
from multitask_personalization.utils import (
    rotation_matrix_x,
    rotation_matrix_y,
    rotmat2euler,
)


class ROMModel(abc.ABC):
    """Base class for ROM models."""

    def __init__(self) -> None:
        self._reachable_points: list[NDArray] = []
        self._reachable_kd_tree: KDTree = KDTree(np.array([[0, 0]]))

    @abc.abstractmethod
    def get_reachable_joints(self) -> NDArray:
        """Get the reachable joints."""

    @abc.abstractmethod
    def get_reachable_points(self) -> list[NDArray]:
        """Get the reachable points."""

    @abc.abstractmethod
    def check_position_reachable(self, position: NDArray) -> bool:
        """Check if a position is reachable."""

    @abc.abstractmethod
    def sample_reachable_position(self, rng: np.random.Generator) -> NDArray:
        """Sample a reachable position."""


class GroundTruthROMModel(ROMModel):
    """ROM model constructed from ground truth data."""

    def __init__(
        self,
        human_spec: HumanSpec,
        ik_distance_threshold: float = 1e-1,
        seed: int = 0,
    ) -> None:
        super().__init__()
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
        print(
            f"Loaded {len(self._reachable_joints)} points"
            + f" from {self._subject}_{self._condition}_dense_points.pkl"
        )
        # Create human for forward IK.
        # Uncomment for debugging
        # from pybullet_helpers.gui import create_gui_connection
        # self._physics_client_id = create_gui_connection()
        self._physics_client_id = p.connect(p.DIRECT)
        self._rng = np.random.default_rng(seed)
        self._human = create_human_from_spec(
            human_spec, self._rng, self._physics_client_id
        )
        # Load reachable points from cache if available, otherwise generate and cache
        cache_dir = "src/multitask_personalization/rom/cache"
        reachable_points_cache_path = (
            f"{cache_dir}/{human_spec.subject_id}_{human_spec.condition}_"
            + f"{human_spec.gender}_{human_spec.impairment}_reachable_points.pkl"
        )

        if os.path.exists(reachable_points_cache_path):
            with open(reachable_points_cache_path, "rb") as f:
                self._reachable_points = pickle.load(f)
            print("Loaded reachable points from cache.")
        else:
            # Create reachable point cloud using human FK.
            self._reachable_points = [
                self._run_human_fk(point) for point in self._reachable_joints
            ]
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            # Cache reachable points
            with open(
                reachable_points_cache_path,
                "wb",
            ) as f:
                pickle.dump(self._reachable_points, f)
            print("Cached reachable points.")
        self._reachable_kd_tree = KDTree(self._reachable_points)

        # Uncomment for debugging.
        # self._visualize_reachable_points()

    def get_reachable_joints(self) -> NDArray:
        return self._reachable_joints

    def get_reachable_points(self) -> list[NDArray]:
        return self._reachable_points

    def check_position_reachable(self, position: NDArray) -> bool:
        distance, _ = self._reachable_kd_tree.query(position)
        return distance < self._ik_distance_threshold

    def sample_reachable_position(self, rng: np.random.Generator) -> NDArray:
        return rng.choice(self._reachable_points)

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
                self.get_reachable_points()[i]
                for i in self._rng.choice(
                    len(self.get_reachable_points()), n, replace=False
                )
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
