"""Tests for rom/models.py."""

import numpy as np
import pybullet as p

from multitask_personalization.envs.pybullet.pybullet_human import (
    create_human_from_spec,
)
from multitask_personalization.envs.pybullet.pybullet_scene_spec import (
    PyBulletSceneSpec,
)
from multitask_personalization.rom.models import SphericalROMModel
from multitask_personalization.utils import (
    sample_on_sphere,
    sample_within_sphere,
)


def test_spherical_rom_model():
    """Tests for rom/models.py."""
    seed = 123

    rng = np.random.default_rng(seed)

    scene_spec = PyBulletSceneSpec()
    sphere_radius = 1.0
    spherical_rom_model = SphericalROMModel(
        scene_spec.human_spec,
        seed=seed,
        min_possible_radius=sphere_radius - 1e-1,
        max_possible_radius=sphere_radius + 1e-1,
    )

    # Create human.
    physics_client_id = p.connect(p.DIRECT)
    human = create_human_from_spec(scene_spec.human_spec, rng, physics_client_id)
    sphere_center = (
        spherical_rom_model._sphere_center
    )  # pylint: disable=protected-access

    # Test SphereROMModel().
    assert spherical_rom_model.check_position_reachable(sphere_center)
    assert spherical_rom_model.sample_reachable_position(rng).shape == (3,)
    # Check check_position_reachable().
    for _ in range(100):
        point = sample_within_sphere(sphere_center, sphere_radius, rng)
        distance = np.linalg.norm(point - sphere_center)
        assert spherical_rom_model.check_position_reachable(point)
    # Check sample_reachable_position().
    for _ in range(100):
        point = spherical_rom_model.sample_reachable_position(rng)
        distance = np.linalg.norm(point - sphere_center)
        assert distance < sphere_radius + 1e-6
    # Check get_position_reachable_logprob().
    for _ in range(100):
        point = sample_within_sphere(sphere_center, sphere_radius - 2e-1, rng)
        assert np.isclose(
            spherical_rom_model.get_position_reachable_logprob(point), 0.0
        )
    for _ in range(100):
        point = sample_on_sphere(sphere_center, sphere_radius + 2e-1, rng)
        assert np.isneginf(spherical_rom_model.get_position_reachable_logprob(point))
    for _ in range(100):
        point = sample_on_sphere(sphere_center, sphere_radius, rng)
        assert np.isclose(
            spherical_rom_model.get_position_reachable_logprob(point), np.log(0.5)
        )

    # Test training the parameters.
    init_params = spherical_rom_model.get_trainable_parameters()
    assert np.isclose(init_params[0], sphere_radius - 1e-1)
    assert np.isclose(init_params[1], sphere_radius + 1e-1)

    data = [
        (sphere_center + np.array([0.0, 0.0, 0.9]), True),
        (sphere_center + np.array([0.0, 0.0, 1.1]), False),
    ]

    spherical_rom_model.train(data)

    new_params = spherical_rom_model.get_trainable_parameters()
    assert np.isclose(new_params[0], 0.9)
    assert np.isclose(new_params[1], 1.1)
