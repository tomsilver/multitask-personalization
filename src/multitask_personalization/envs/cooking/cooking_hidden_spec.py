"""Hidden specification for the cooking environment."""


from __future__ import annotations

from dataclasses import dataclass, field



@dataclass(frozen=True)
class CookingHappyMeal:
    """A specification of one of the meals that the user enjoys."""

    # Ingredient name to min/max bounds.
    ingredients: dict[str, tuple[float, float]]

    def __post_init__(self) -> None:
        for lower, upper in self.ingredients.values():
            assert lower < upper


@dataclass(frozen=True)
class CookingHiddenSpec:
    """Hidden parameters for a cooking environment."""

    happy_meals: list[CookingHappyMeal] = field(
        default_factory=lambda : [
            CookingHappyMeal({
                "salt": (0.1, 0.2),
                "pepper": (0.2, 0.3),
            }),
            CookingHappyMeal({
                "salt": (0.1, 0.2),
                "sugar": (0.2, 0.3),
                "flour": (1.0, 1.2),
            }),
        ]
    )
