"""A name typed without capitals is still read, and only when it is a name.

Both English name readers keyed on capitalisation — the regex required
``[A-Z]`` and spaCy's PERSON model relies on case — so "send 50 to ahmed" used
to lose the beneficiary entirely and the assistant asked "who to?" about a name
the customer had already typed. The rescue pass must not invent a payee out of
the words that follow "to" in a question.
"""

import pytest

from app.nlu import english
from app.nlu.entities import extract_lowercase_recipient, extract_recipient
from app.schemas import Language


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("send 50 to ahmed", "ahmed"),
        ("plz transfer 800 to noura saad", "noura saad"),
        ("hey can u send 60 to laila", "laila"),
        ("transfer 1200 sar to khalid fahad", "khalid fahad"),
        ("send 500 to ahmed from savings", "ahmed"),
        ("transfer money to my brother ahmed", "ahmed"),
        ("i want to send money to sara adel", "sara adel"),
        ("send two hundred to mona", "mona"),
    ],
)
def test_lowercase_name_is_read(text: str, expected: str) -> None:
    assert extract_recipient(text, Language.EN) == expected


@pytest.mark.parametrize(
    "text",
    [
        "transfer 500 to my savings account",
        "send 250 to account number 8899",
        "pay 120 to the electricity bill",
        "convert 500 sar to usd",
        "transfer 90 to iban sa1122330000001200",
        "send 60 to me",
        "i sent 500 to the wrong person what do i do",
        "i need to talk to customer service",
        "how do i send money to someone new",
        "i transferred 300 to the same person twice",
        "what happened to my last transfer",
        "i want to add a beneficiary",
        # memory answers this one from the user's habits
        "send 20 usd to my usual",
    ],
)
def test_a_question_never_produces_a_payee(text: str) -> None:
    assert extract_lowercase_recipient(text) is None
    assert extract_recipient(text, Language.EN) is None


def test_the_customers_spelling_is_kept_verbatim() -> None:
    """No gazetteer correction here: the directory decides, not a fuzzy score."""

    assert extract_recipient("send 70 to mohamd nour", Language.EN) == "mohamd nour"
    assert extract_recipient("send 70 to noura saad", Language.EN) == "noura saad"


def test_capitalised_names_are_unaffected() -> None:
    assert english.extract_entities("send 50 to Ahmed").recipient == "Ahmed"
    assert (
        english.extract_entities("wire 3000 USD to Ahmed Mahmoud").recipient
        == "Ahmed Mahmoud"
    )


def test_arabic_is_untouched() -> None:
    assert extract_recipient("ارسل 50 لأحمد", Language.AR) == "أحمد"
    assert extract_recipient("كم رصيدي", Language.AR) is None
