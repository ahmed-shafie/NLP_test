"""Three ways the assistant used to answer something the customer never asked.

Observed in a live session:

* while it waited for a biller, every reply became the biller's name — "is the
  data shared with others" came back as a company that isn't in the catalogue;
* "قدم على قرض" and "افتح محفظة استثمارية" were answered with the customer's
  balance, because a product application has no intent of its own and lands on
  whichever one sits nearest in embedding space;
* "full amount" was not read as an answer to "how much?" at all.

The figure behind "the full amount" is the point of the last group: a bill's
outstanding amount is not readable here, so it is refused rather than invented,
and a transfer's is the Banking Core's balance, offered for an explicit yes.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app import banking_core_client as bcc
from app.conversation.engine import ConversationEngine
from app.conversation.products import Product, requested_product
from app.conversation.reasons import ReasonCode
from app.conversation.state import ConversationStatus
from app.schemas import Intent, Language


@pytest.fixture()
def engine() -> ConversationEngine:
    return ConversationEngine()


@pytest.fixture()
def core_balance(monkeypatch: pytest.MonkeyPatch) -> Decimal:
    """A Banking Core that states one spendable balance."""

    balance = Decimal("1234.50")
    monkeypatch.setattr(
        bcc,
        "get_balance",
        lambda owner_user, account=None, account_type=None: bcc.AccountInfo(
            account_id="ACC-1",
            account_type="current",
            number="SA0380009999888877",
            currency="SAR",
            balance=balance,
            status="active",
        ),
    )
    return balance


# --------------------------------------------------------------------------- #
# The biller prompt accepts a biller, not a sentence.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "reply",
    [
        "is the data shared with others",
        "do you work at anb",
        "who are you",
        "activate a beneficiary",
        "كم درجة الحرارة",
    ],
)
def test_a_sentence_is_not_quoted_back_as_a_biller(
    engine: ConversationEngine, reply: str
) -> None:
    opening = engine.handle("pay a bill", session_id=f"biller-{reply}")
    assert opening.state.pending_slot == "biller"

    result = engine.handle(reply, session_id=f"biller-{reply}")

    assert reply not in result.reply
    assert result.reason is ReasonCode.INVALID_SLOT_VALUE
    assert result.state.slots.biller is None
    assert result.state.intent is Intent.PAY_BILL
    assert result.state.pending_slot == "biller"


def test_the_biller_prompt_still_takes_a_biller_after_a_bad_answer(
    engine: ConversationEngine,
) -> None:
    engine.handle("pay a bill", session_id="biller-recover")
    engine.handle("who are you", session_id="biller-recover")

    result = engine.handle("stc", session_id="biller-recover")

    assert result.state.slots.biller == "STC"
    assert result.state.pending_slot == "reference_number"


def test_a_named_biller_outside_the_catalogue_is_still_named(
    engine: ConversationEngine,
) -> None:
    result = engine.handle("ادفع فاتورة سبتكو", session_id="biller-named")

    assert "سبتكو" in result.reply
    assert result.reason is ReasonCode.BILLER_NOT_IN_CATALOGUE


# --------------------------------------------------------------------------- #
# Products this assistant does not open.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "language", "product"),
    [
        ("apply a loan", Language.EN, Product.LOAN),
        ("i want a credit card", Language.EN, Product.CARD),
        ("قدم على قرض", Language.AR, Product.LOAN),
        ("اطلب بطاقة ايتمانية", Language.AR, Product.CARD),
        ("افتح محفظة استثمارية", Language.AR, Product.INVESTMENT),
        ("open a new account", Language.EN, Product.ACCOUNT),
    ],
)
def test_an_application_is_recognised_as_one(
    text: str, language: Language, product: Product
) -> None:
    assert requested_product(text, language) is product


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("pay my card bill", Language.EN),
        ("transfer 500 to nasser", Language.EN),
        ("ادفع فاتورة الكهرباء", Language.AR),
        ("كم رصيدي", Language.AR),
        ("حول 200 لمحمد", Language.AR),
    ],
)
def test_a_payment_is_not_read_as_an_application(text: str, language: Language) -> None:
    assert requested_product(text, language) is None


@pytest.mark.parametrize(
    "text",
    ["apply a loan", "قدم على قرض", "افتح محفظة استثمارية", "اطلب بطاقة ايتمانية"],
)
def test_an_application_is_refused_without_disclosing_a_balance(
    engine: ConversationEngine, core_balance: Decimal, text: str
) -> None:
    result = engine.handle(text, session_id=f"product-{text}")

    assert result.reason is ReasonCode.PRODUCT_NOT_SUPPORTED
    assert str(core_balance) not in result.reply
    assert result.state.intent is None
    assert result.state.pending_slot is None
    assert result.state.status is ConversationStatus.SELECTING


def test_an_application_mid_transfer_leaves_the_transfer_standing(
    engine: ConversationEngine,
) -> None:
    engine.handle("transfer to nasser", session_id="product-midflow")

    result = engine.handle("i want a credit card", session_id="product-midflow")

    assert result.reason is ReasonCode.PRODUCT_NOT_SUPPORTED
    assert result.state.intent is Intent.TRANSFER_MONEY
    assert result.state.pending_slot == "amount"
    assert result.state.slots.amount is None


# --------------------------------------------------------------------------- #
# "The full amount".
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text", ["full amount", "the full amount", "كامل المبلغ"])
def test_a_bill_refuses_an_amount_it_cannot_read(
    engine: ConversationEngine, text: str
) -> None:
    engine.handle("ادفع فاتورة stc", session_id=f"whole-bill-{text}")
    engine.handle("4321", session_id=f"whole-bill-{text}")

    result = engine.handle(text, session_id=f"whole-bill-{text}")

    assert result.reason is ReasonCode.AMOUNT_DUE_UNAVAILABLE
    assert result.state.slots.amount is None
    assert result.state.pending_slot == "amount"
    assert result.state.slots.reference_number == "4321"


def test_a_bill_still_takes_a_figure_after_refusing_the_full_amount(
    engine: ConversationEngine,
) -> None:
    engine.handle("ادفع فاتورة stc", session_id="whole-bill-then-figure")
    engine.handle("4321", session_id="whole-bill-then-figure")
    engine.handle("full amount", session_id="whole-bill-then-figure")

    result = engine.handle("300", session_id="whole-bill-then-figure")

    assert result.state.slots.amount == Decimal("300")


def test_a_transfer_offers_the_balance_the_core_states(
    engine: ConversationEngine, core_balance: Decimal
) -> None:
    engine.handle("transfer to nasser", session_id="whole-transfer")

    result = engine.handle("full amount", session_id="whole-transfer")

    assert "1234.5" in result.reply
    assert result.state.offered_amount == core_balance
    assert result.state.slots.amount is None
    assert result.state.pending_slot == "amount"


def test_the_offered_balance_becomes_the_amount_only_on_a_yes(
    engine: ConversationEngine, core_balance: Decimal
) -> None:
    engine.handle("transfer to nasser", session_id="whole-transfer-yes")
    engine.handle("full amount", session_id="whole-transfer-yes")

    result = engine.handle("yes", session_id="whole-transfer-yes")

    assert result.state.slots.amount == core_balance
