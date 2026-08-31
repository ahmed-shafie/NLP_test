"""Bank products this assistant cannot open, and how to recognise a request for one.

The assistant does three things: it answers a balance, it transfers money, and it
pays a bill. Everything else a bank sells — a loan, a card, an investment wallet,
a new account — is an application, not a payment, and it is opened somewhere else.

Left unrecognised, those requests fall through to whichever intent sits nearest in
embedding space, which is how "افتح محفظة استثمارية" came back as a balance: an
answer about the customer's money to a question that never asked for it. Naming
the product explicitly is both a truthful answer and a refusal to open a flow.

A product is only *requested* when the message names one and asks to acquire it,
and never when the message carries payment evidence of its own — an amount, a
payee, a biller or a bill — because a real payment mentioning a card ("pay my
card bill") is a payment, not an application.
"""

from __future__ import annotations

from enum import Enum

from app.nlu import entities
from app.nlu.normalize import normalize, strip_proclitic
from app.schemas import Language


class Product(str, Enum):
    """A product the customer can ask for and this assistant cannot open."""

    LOAN = "loan"
    CARD = "card"
    INVESTMENT = "investment"
    ACCOUNT = "account"


def _keys(words: tuple[str, ...]) -> frozenset[str]:
    """Match words, their normalised form, and their form without the article."""

    normalized = {normalize(word) for word in words}
    return frozenset(normalized | {strip_proclitic(word) for word in normalized})


_PRODUCT_NOUNS: dict[Product, frozenset[str]] = {
    Product.LOAN: _keys(("loan", "financing", "mortgage", "قرض", "تمويل", "سلفة")),
    Product.CARD: _keys(("card", "بطاقة", "بطاقه")),
    Product.INVESTMENT: _keys(
        ("investment", "portfolio", "استثمار", "استثمارية", "محفظة", "أسهم", "اسهم")
    ),
    Product.ACCOUNT: _keys(("account", "حساب", "محفظة")),
}

# Verbs that ask to be *given* the product, rather than to act on an existing one.
_ACQUIRE_VERBS = _keys(
    (
        "apply",
        "issue",
        "open",
        "request",
        "want",
        "need",
        "get",
        "give",
        "order",
        "قدم",
        "أقدم",
        "اطلب",
        "أطلب",
        "افتح",
        "أفتح",
        "اصدر",
        "أصدر",
        "ابغى",
        "أبغى",
        "ابي",
        "أبي",
        "احتاج",
        "أحتاج",
        "عايز",
        "ودي",
    )
)

# A product application is one order among many; the widest reading wins so that
# "a credit card and an investment account" is answered about the card.
_PRIORITY = (Product.LOAN, Product.CARD, Product.INVESTMENT, Product.ACCOUNT)


def _tokens(text: str) -> set[str]:
    words = {word.strip(".,!?؟،:؛\"'()") for word in normalize(text).split()}
    return {word for word in words if word} | {
        strip_proclitic(word) for word in words if word
    }


def requested_product(text: str, language: Language) -> Product | None:
    """The product this message asks to be given, if it asks for one.

    Payment evidence vetoes the reading: a message that names an amount, a payee,
    a biller or a bill is a payment the money flows already handle, whatever
    product word it happens to mention.
    """

    tokens = _tokens(text)
    if not tokens & _ACQUIRE_VERBS:
        return None
    if entities.extract_amount(text) is not None:
        return None
    if entities.has_bill_word(text):
        return None
    bills = entities.extract_bill_entities(text, language)
    if bills.biller is not None or bills.reference_number is not None:
        return None
    if entities.extract_recipient(text, language) is not None:
        return None
    for product in _PRIORITY:
        if tokens & _PRODUCT_NOUNS[product]:
            return product
    return None
