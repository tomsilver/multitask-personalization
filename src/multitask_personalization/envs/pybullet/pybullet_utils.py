

import numpy as np
from tomsutils.llm import OpenAILLM


def get_user_book_enjoyment_logprob(
    book_description: str,
    user_preferences: str,
    llm: OpenAILLM,
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
    llm: OpenAILLM,
    seed: int = 0,
) -> float:
    """Return whether the user would enjoy the book."""
    lp = get_user_book_enjoyment_logprob(book_description, user_preferences, llm, seed)
    return lp > np.log(0.5)