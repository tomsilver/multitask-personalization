"""Hidden spec for assistive feeding environment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeedingHiddenSceneSpec:
    """Defines hidden parameters for the assitive feeding environment."""

    # A number between 0 and 1, where 0 means that the user doesn't care at all
    # about occlusion, and 1 means they care a lot. We define a fixed line of
    # sight "body" (e.g., a cuboid emanating from the user's face) and then scale
    # that body by this factor and then check for collisions between the body
    # and the robot.
    occlusion_preference_scale: float = 0.95
