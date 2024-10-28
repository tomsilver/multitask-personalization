"""Tests for rom/models.py."""

import numpy as np
import pybullet as p

from multitask_personalization.envs.pybullet.pybullet_human_spec import (
    create_human_from_spec,
)
from multitask_personalization.envs.pybullet.pybullet_task_spec import (
    PyBulletTaskSpec,
)
from multitask_personalization.rom.models import SphericalROMModel
from multitask_personalization.utils import (
    sample_spherical,
)


def test_spherical_rom_model():
    """Tests for rom/models.py."""
    seed = 123

    rng = np.random.default_rng(seed)

    task_spec = PyBulletTaskSpec()
    sphere_radius = rng.uniform(0.1, 0.5)
    spherical_rom_model = SphericalROMModel(
        task_spec.human_spec, seed=seed, radius=sphere_radius
    )

    # Create human
    physics_client_id = p.connect(p.DIRECT)
    human = create_human_from_spec(task_spec.human_spec, rng, physics_client_id)
    sphere_center, _ = human.get_pos_orient(human.right_wrist)

    # Test SphereROMModel
    assert spherical_rom_model.check_position_reachable(sphere_center)
    assert spherical_rom_model.sample_reachable_position(rng).shape == (3,)
    # Check check_position_reachable
    for _ in range(100):
        point = sample_spherical(sphere_center, sphere_radius, rng)
        distance = np.linalg.norm(point - sphere_center)
        assert spherical_rom_model.check_position_reachable(point)
    # Check sample_reachable_position
    for _ in range(100):
        point = spherical_rom_model.sample_reachable_position(rng)
        distance = np.linalg.norm(point - sphere_center)
        assert distance < sphere_radius + 1e-6
