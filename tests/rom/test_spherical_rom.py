"""Tests for rom/models.py."""

import numpy as np

from multitask_personalization.envs.pybullet.pybullet_scene_spec import (
    PyBulletSceneSpec,
)
from multitask_personalization.rom.models import SphericalROMModel
from multitask_personalization.utils import (
    sample_within_sphere,
)


def test_spherical_rom_model():
    """Tests for rom/models.py."""
    seed = 123

    rng = np.random.default_rng(seed)

    scene_spec = PyBulletSceneSpec()
    min_radius = 0.4
    max_radius = 0.6
    spherical_rom_model = SphericalROMModel(
        scene_spec.human_spec,
        seed=seed,
        min_possible_radius=0,
        max_possible_radius=100,
    )
    sphere_center = (
        spherical_rom_model._sphere_center  # pylint: disable=protected-access
    )
    data = [
        (sphere_center + np.array([0.0, 0.0, min_radius - 1e-2]), False),
        (sphere_center + np.array([0.0, 0.0, min_radius]), True),
        (sphere_center + np.array([0.0, 0.0, max_radius]), True),
        (sphere_center + np.array([0.0, 0.0, max_radius + 1e-2]), False),
    ]
    spherical_rom_model.train(data)

    assert not spherical_rom_model.check_position_reachable(sphere_center)
    for position, label in data:
        lp = spherical_rom_model.get_position_reachable_logprob(position)
        if label:
            assert np.isclose(lp, 0.0)
        else:
            assert np.isneginf(lp)
    test_point = sphere_center + np.array([0.0, 0.0, min_radius - 5e-3])
    test_lp = spherical_rom_model.get_position_reachable_logprob(test_point)
    assert np.isclose(test_lp, np.log(0.5))
    test_point = sphere_center + np.array([0.0, 0.0, max_radius + 5e-3])
    test_lp = spherical_rom_model.get_position_reachable_logprob(test_point)
    assert np.isclose(test_lp, np.log(0.5))

    assert spherical_rom_model.sample_reachable_position(rng).shape == (3,)

    # Check check_position_reachable().
    for _ in range(100):
        point = sample_within_sphere(sphere_center, min_radius, max_radius, rng)
        assert spherical_rom_model.check_position_reachable(point)

    # Check sample_reachable_position().
    for _ in range(100):
        point = spherical_rom_model.sample_reachable_position(rng)
        distance = np.linalg.norm(np.subtract(point, sphere_center))
        assert min_radius - 5e-3 <= distance <= max_radius + 5e-3
