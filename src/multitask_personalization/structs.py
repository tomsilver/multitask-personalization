"""Data structures."""

from functools import cached_property
from typing import (
    Any,
    Generic,
    Hashable,
    Iterator,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

# Type aliases.
Image: TypeAlias = NDArray[np.uint8]


class Hyperparameters:
    """General hyperparameters."""


_T = TypeVar("_T", bound="HashableComparable")


@runtime_checkable
class HashableComparable(Protocol):
    """Used for type checking objects that must be hashable and comparable."""

    def __hash__(self) -> int: ...

    def __eq__(self, other: Any) -> bool: ...

    def __lt__(self: _T, other: _T) -> bool: ...


class CategoricalDistribution(Generic[_T]):
    """A categorical distribution."""

    def __init__(
        self, outcome_to_prob: dict[_T, float], normalize: bool = False
    ) -> None:
        # Prune zero entries and avoid modifying input.
        d = {o: np.log(p) for o, p in outcome_to_prob.items() if p > 0}
        # Normalize the distribution if asked.
        if normalize:
            z = logsumexp(list(d.values()))
            d = {o: lp - z for o, lp in d.items()}
        # Finalize.
        self._outcome_to_log_prob = d

    @cached_property
    def _hashable(self) -> Hashable:
        # NOTE: it is important to sort here, even though dictionary ordering
        # is deterministic, because the same distribution may be created with
        # two different dictionary orderings.
        return tuple(
            sorted(
                (o, np.round(np.exp(lp), decimals=6))
                for o, lp in self._outcome_to_log_prob.items()
            )
        )

    @cached_property
    def _hash(self) -> int:
        return hash(self._hashable)

    @cached_property
    def _str(self) -> str:
        return f"CategoricalDistribution({self._hashable})"

    def __hash__(self) -> int:
        return self._hash

    def __call__(self, outcome: _T) -> float:
        """Get the probability for the given outcome."""
        return self[outcome]

    def __getitem__(self, outcome: _T) -> float:
        log_prob = self._outcome_to_log_prob.get(outcome, -np.inf)
        prob = np.exp(log_prob)
        assert 0 <= prob <= 1
        return prob

    def __iter__(self) -> Iterator[_T]:
        return iter(self._outcome_to_log_prob)

    def __repr__(self) -> str:
        return self._str

    def __str__(self) -> str:
        return self._str

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, CategoricalDistribution)
        outcomes = set(self) | set(other)
        return all(np.isclose(self(o), other(o)) for o in outcomes)

    def __lt__(self, other: Any) -> bool:
        assert isinstance(other, CategoricalDistribution)
        return str(self) < str(other)

    def items(self) -> Iterator[tuple[_T, float]]:
        """Iterate the dictionary."""
        for outcome, log_prob in self._outcome_to_log_prob.items():
            yield (outcome, np.exp(log_prob))

    def sample(self, rng: np.random.Generator) -> _T:
        """Draw a random sample."""
        candidates = sorted(self)
        log_probs = [self[c] for c in candidates]
        # https://stackoverflow.com/questions/58339083
        n = len(candidates)
        gumbels = rng.gumbel(size=n)
        idx = np.argmax(gumbels + log_probs)
        return candidates[idx]
