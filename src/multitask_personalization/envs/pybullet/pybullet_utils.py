"""Utilities for the pybullet environment and models."""

import re

import numpy as np
import PIL
from pybullet_helpers.geometry import Pose
from tomsutils.llm import LargeLanguageModel

BANISH_POSE = Pose((-1000, -1000, -1000))


def get_user_book_enjoyment_logprob(
    book_description: str,
    user_preferences: str,
    llm: LargeLanguageModel,
    seed: int = 0,
    num_bins: int = 11,
) -> float:
    """Return a logprob that the user would enjoy the book."""

    prompt = f"""Book description: {book_description}

User description: {user_preferences}

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
    return lp >= np.log(0.5) - 1e-6


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

        if prompt.startswith("Generate a list of "):
            # pylint: disable=line-too-long
            pattern = r"Generate a list of (\d+) real English-language book titles and authors"
            matches = re.findall(pattern, prompt)
            assert len(matches) == 1
            num_books = int(matches[0])
            if num_books == 0:
                return []
            assert num_books >= 1
            # NOTE: it's important to not randomize here because it doesn't make
            # sense to test generalization over book numbers! For example, if
            # we randomized, the robot could be evaluated on never-before-seen
            # book numbers, and there's no way it could do anything useful.
            num_loved_books = 2
            books: list[str] = []
            for i in range(num_books):
                if i < num_loved_books:
                    title = (
                        f"{i+1}. [The user would love] Title: Book {i}. Author: Love."
                    )
                else:
                    title = (
                        f"{i+1}. [The user would hate] Title: Book {i}. Author: Hate."
                    )
                books.append(title)
            resp = "\n".join(books)
            return [resp]

        if prompt.startswith(
            "Pretend you are a human user with the following preferences"
        ):
            book_number = self._get_book_number_from_description(prompt)
            resp = f"<SEEN>{book_number}</SEEN>"
            return [resp]

        if prompt.startswith("Below is a first-person history of interactions"):
            pattern = r"<SEEN>(\d+)</SEEN>"
            matches = re.findall(pattern, prompt)
            numbers = [int(num) for num in matches]
            resp = " ".join([f"<SEEN>{i}</SEEN>" for i in numbers])
            return [resp]

        raise NotImplementedError

    def get_multiple_choice_logprobs(
        self, prompt: str, choices: list[str], seed: int
    ) -> dict[str, float]:
        book_description, _, user_description, remainder = prompt.split("\n", 3)
        assert book_description.startswith("Book description: ")
        assert user_description.startswith("User description: ")
        assert "0 to 10" in remainder
        book_number = self._get_book_number_from_description(book_description)

        if "Author: Love" in book_description:
            love_or_hate = "love"
        else:
            assert "Author: Hate" in book_description
            love_or_hate = "hate"

        # Book is either unknown, known hate, or known love.
        status = "unknown"
        if f"<SEEN>{book_number}</SEEN>" in user_description:
            status = love_or_hate

        # If the user description is a ground-truth description then the status
        # is known. For example, "I enjoy some fiction, ..."
        if not ("<SEEN>" in user_description or "Unknown" in user_description):
            status = love_or_hate

        if status == "unknown":
            response_num = 5
        elif status == "love":
            response_num = 10
        else:
            assert status == "hate"
            response_num = 0

        logprobs = {str(i): -np.inf for i in choices}
        logprobs[str(response_num)] = 0.0
        return logprobs

    def _get_book_number_from_description(self, description: str) -> int:
        prefix = "Book description: Title: Book "
        assert prefix in description
        after_prefix = description[description.index(prefix) + len(prefix) :]
        book_number = int(after_prefix[: after_prefix.index(".")])
        return book_number
