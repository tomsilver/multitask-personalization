"""Hidden specification for the cooking environment."""


from __future__ import annotations

from dataclasses import dataclass, field



@dataclass(frozen=True)
class CookingHappyMeal:
    """A specification of one of the meals that the user enjoys."""

    # Ingredient name, temperature bounds, quantity bounds.
    ingredients: list[tuple[str, tuple[float, float], tuple[float, float]]]

    def __post_init__(self) -> None:
        for _, (temp_lo, temp_hi), (quant_lo, quant_hi) in self.ingredients:
            assert 0 <= temp_lo < temp_hi
            assert 0 <= quant_lo < quant_hi


@dataclass(frozen=True)
class CookingHiddenSpec:
    """Hidden parameters for a cooking environment."""

    happy_meals: list[CookingHappyMeal] = field(
        default_factory=lambda : [
            CookingHappyMeal([
                ("salt", (0.1, 0.2), (0.0, 1.0)),
                ("pepper", (0.2, 0.3), (0.0, 1.0)),
            ]),
            CookingHappyMeal([
                ("salt", (0.1, 0.2), (0.0, 1.0)),
                ("sugar", (0.2, 0.3), (0.0, 1.0)),
                ("flour", (1.0, 1.2), (0.0, 1.0)),
            ]),
        ]
    )
