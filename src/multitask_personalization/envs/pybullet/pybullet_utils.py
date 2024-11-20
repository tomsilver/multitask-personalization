"""Utilities for the pybullet environment and models."""

import numpy as np
import PIL
from tomsutils.llm import LargeLanguageModel


def get_user_book_enjoyment_logprob(
    book_description: str,
    user_preferences: str,
    llm: LargeLanguageModel,
    seed: int = 0,
    num_bins: int = 11,
) -> float:
    """Return a logprob that the user would enjoy the book."""

    prompt = f"""Book description: {book_description}

User description: {user_preferences}.

How much would the user enjoy the book on a scale from 0 to {num_bins-1}, where 0 means hate and {num_bins-1} means love?
"""
    choices = [str(i) for i in range(num_bins)]
    logprobs = llm.get_multiple_choice_logprobs(prompt, choices, seed)
    # Interpretation: 8 out of 10 means that 8 times out of 10, the user would
    # report liking it; 2 times out of 10, they would report not liking it.
    expectation = 0.0
    for i in range(num_bins):
        enjoy_prob = i / (num_bins - 1)
        logprob = logprobs[str(i)]
        expectation += enjoy_prob * np.exp(logprob)
    return np.log(expectation)


def user_would_enjoy_book(
    book_description: str,
    user_preferences: str,
    llm: LargeLanguageModel,
    seed: int = 0,
) -> float:
    """Return whether the user would enjoy the book."""
    lp = get_user_book_enjoyment_logprob(book_description, user_preferences, llm, seed)
    return lp > np.log(0.5)


class PyBulletCannedLLM(LargeLanguageModel):
    """A very domain-specific 'LLM' that we can use for testing things without
    paying any money to OpenAI."""

    def get_id(self) -> str:
        return "pybullet-canned"

    def _sample_completions(
        self,
        prompt: str,
        imgs: list[PIL.Image.Image] | None,
        temperature: float,
        seed: int,
        num_completions: int = 1,
    ) -> list[str]:

        assert num_completions == 1
        rng = np.random.default_rng(seed)

        if prompt.startswith("Generate a list of 3 real English-language book titles and authors."):
            books = [
                f"1. [The user would love] Title: Book {rng.integers(1, 101)}. Author: Love.",
                f"2. [The user would hate] Title: Book {rng.integers(1, 101)}. Author: Hate.",
                f"3. [The user would hate] Title: Book {rng.integers(1, 101)}. Author: Hate.",
            ]
            resp = "\n".join(books)
            return [resp]

        else:
            import ipdb; ipdb.set_trace()
            raise NotImplementedError

    def get_multiple_choice_logprobs(
        self, prompt: str, choices: list[str], seed: int
    ) -> dict[str, float]:
        book_description, _, user_description, remainder = prompt.split("\n", 3)
        assert book_description.startswith("Book description: ")
        assert user_description.startswith("User description: ")
        assert "0 to 10" in remainder
        
        if "Unknown" in user_description:
            logprobs = {i: -np.inf for i in choices}
            logprobs[5] = 0.0
            return logprobs
        
        else:
            import ipdb; ipdb.set_trace()
            raise NotImplementedError

