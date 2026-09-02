"""The two-tier reply split: money replies deterministic, the rest fluent.

The load-bearing tests here are the architectural ones: a money-critical reply must
be unreachable from the rewrite path, and the guard must reject any rewrite that
touches a number or a code. Everything else in this file is fluency polish.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import pytest

from app.config import settings
from app.conversation import phrasing, templates
from app.schemas import Language

# Not a reply: a cue detector used by the router.
_NOT_A_REPLY = {"is_small_talk"}


def _reply_functions() -> dict[str, Callable[..., str]]:
    return {
        name: fn
        for name, fn in vars(templates).items()
        if inspect.isfunction(fn)
        and fn.__module__ == templates.__name__
        and not name.startswith("_")
        and name not in _NOT_A_REPLY
    }


def test_every_reply_declares_a_tier() -> None:
    """A new reply must be classified; forgetting to do so fails here, not in prod."""

    for name in _reply_functions():
        assert phrasing.tier_of(name) in (
            phrasing.Tier.CRITICAL,
            phrasing.Tier.CONVERSATIONAL,
        ), name


def test_tiers_are_disjoint_and_cover_only_real_replies() -> None:
    assert not (phrasing.CRITICAL_REPLIES & phrasing.CONVERSATIONAL_REPLIES)
    declared = phrasing.CRITICAL_REPLIES | phrasing.CONVERSATIONAL_REPLIES
    assert declared == set(_reply_functions())


def test_unknown_reply_has_no_tier() -> None:
    with pytest.raises(KeyError):
        phrasing.tier_of("something_new")


@pytest.mark.parametrize("key", sorted(phrasing.CRITICAL_REPLIES))
def test_money_critical_replies_cannot_be_rewritten(key: str) -> None:
    """The tier is a call-site guarantee, not a convention."""

    with pytest.raises(ValueError, match="money-critical"):
        phrasing.rewrite(key, "Just to confirm — send 500 SAR to Ahmed?", Language.EN)


@pytest.mark.parametrize("key", sorted(phrasing.CRITICAL_REPLIES))
def test_money_critical_templates_never_touch_the_phrasing_layer(key: str) -> None:
    """No critical reply may route through variation or rewriting."""

    source = inspect.getsource(_reply_functions()[key])
    assert "phrasing." not in source


def test_money_replies_are_unchanged_while_a_model_is_rewriting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With rewriting on and a hostile model installed, money wording still holds."""

    monkeypatch.setattr(settings, "reply_rewrite_enabled", True)
    monkeypatch.setattr(
        phrasing,
        "_CACHE",
        {},
    )

    class HostileHandler:
        def rephrase(self, text: str, language: str, timeout: float) -> str:
            return "All done! I sent about five hundred riyals to whoever you meant."

    monkeypatch.setattr("app.llm.get_llm_handler", lambda: HostileHandler())

    assert templates.confirm_prompt("500", "SAR", "Ahmed", Language.EN) == (
        "Just to confirm — send 500 SAR to Ahmed? (yes/no)"
    )
    assert templates.completed("500", "SAR", "Ahmed", Language.EN).startswith(
        "All set!"
    )
    assert "500" in templates.bill_completed("500", "SAR", "STC", "12345", Language.EN)
    assert "SA••7519" in templates.beneficiary_add_completed(
        "Ahmed", "SA••7519", Language.EN
    )


# ------------------------------------------------------------------ the guard


def test_guard_accepts_a_faithful_rewrite() -> None:
    template = "Sure — how much would you like to send?"
    assert phrasing.guard(template, "Sure, how much are we sending?", Language.EN)


def test_guard_rejects_an_invented_number() -> None:
    template = "Sure — how much would you like to send?"
    assert phrasing.guard(template, "Sending 500 now, right?", Language.EN) is None


def test_guard_rejects_an_altered_number() -> None:
    template = "That IBAN is 24 characters."
    assert phrasing.guard(template, "That IBAN is 22 characters.", Language.EN) is None


def test_guard_rejects_arabic_indic_digits_for_a_latin_number() -> None:
    template = "That IBAN is 24 characters."
    assert phrasing.guard(template, "That IBAN is ٢٤ characters.", Language.EN) is None


def test_guard_rejects_a_dropped_code() -> None:
    template = "Sorry, SA0380000000608010167519 failed its checksum."
    assert phrasing.guard(template, "Sorry, that IBAN is wrong.", Language.EN) is None


def test_guard_rejects_an_invented_code() -> None:
    template = "Which bill are we paying?"
    assert phrasing.guard(template, "Paying your STC bill?", Language.EN) is None


def test_guard_rejects_the_wrong_script() -> None:
    template = "تمام — كم المبلغ؟"
    assert phrasing.guard(template, "Sure, how much?", Language.AR) is None
    assert phrasing.guard("Sure, how much?", "كم المبلغ؟", Language.EN) is None


def test_guard_rejects_a_leaked_placeholder() -> None:
    template = "Who should I send it to?"
    assert phrasing.guard(template, "Who is {name}?", Language.EN) is None


def test_guard_rejects_a_rambling_rewrite() -> None:
    template = "Who should I send it to?"
    assert phrasing.guard(template, "Well, " + "so " * 200, Language.EN) is None


def test_guard_rejects_an_empty_rewrite() -> None:
    assert phrasing.guard("Who should I send it to?", "   ", Language.EN) is None


# ---------------------------------------------------------------- the rewrite


def test_a_safe_rewrite_reaches_the_customer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "reply_rewrite_enabled", True)
    monkeypatch.setattr(phrasing, "_CACHE", {})

    class Handler:
        def rephrase(self, text: str, language: str, timeout: float) -> str:
            return "Sure thing — how much are we sending?"

    monkeypatch.setattr("app.llm.get_llm_handler", lambda: Handler())
    assert (
        templates.slot_prompt("amount", Language.EN)
        == "Sure thing — how much are we sending?"
    )


def test_an_unsafe_rewrite_falls_back_to_the_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "reply_rewrite_enabled", True)
    monkeypatch.setattr(phrasing, "_CACHE", {})

    class Handler:
        def rephrase(self, text: str, language: str, timeout: float) -> str:
            return "Sure — shall I send the usual 500?"

    monkeypatch.setattr("app.llm.get_llm_handler", lambda: Handler())
    assert (
        templates.slot_prompt("amount", Language.EN)
        == "Sure — how much would you like to send?"
    )


def test_a_missing_model_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "reply_rewrite_enabled", True)
    monkeypatch.setattr(phrasing, "_CACHE", {})
    monkeypatch.setattr("app.llm.get_llm_handler", lambda: None)
    assert templates.small_talk("hi there", Language.EN).startswith("Hey!")


def test_a_rewrite_is_only_paid_for_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "reply_rewrite_enabled", True)
    monkeypatch.setattr(phrasing, "_CACHE", {})
    calls = []

    class Handler:
        def rephrase(self, text: str, language: str, timeout: float) -> str:
            calls.append(text)
            return "Sure thing — how much are we sending?"

    monkeypatch.setattr("app.llm.get_llm_handler", lambda: Handler())
    for _ in range(5):
        templates.slot_prompt("amount", Language.EN)
    assert len(calls) == 1


def test_rewriting_is_off_by_default() -> None:
    assert settings.reply_rewrite_enabled is False


# -------------------------------------------------------------- the variation


def test_a_conversational_reply_has_several_phrasings_and_no_instant_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "reply_variation_enabled", True)
    monkeypatch.setattr(phrasing, "_LAST_INDEX", {})

    seen = [templates.slot_prompt("amount", Language.AR) for _ in range(20)]
    assert len(set(seen)) > 1
    assert all(a != b for a, b in zip(seen, seen[1:], strict=False))


@pytest.mark.parametrize("language", [Language.EN, Language.AR])
def test_every_conversational_phrasing_bank_is_bilingual_and_plural(
    language: Language,
) -> None:
    banks = [
        templates._SLOT_PROMPTS["amount"],
        templates._SLOT_PROMPTS["recipient"],
        templates._CHOOSE_ACTION,
        templates._FALLBACK,
        templates._GREETING,
        templates._CANCELLED,
        templates._SMALL_TALK["thanks"],
        templates._SMALL_TALK["capability"],
    ]
    for bank in banks:
        variants = bank[language]
        assert len(variants) >= 3
        assert len(set(variants)) == len(variants)


def test_variation_off_pins_the_first_phrasing() -> None:
    """The suite's determinism (and a rollback switch) depends on this."""

    assert (
        templates.slot_prompt("amount", Language.EN)
        == templates._SLOT_PROMPTS["amount"][Language.EN][0]
    )


# ------------------------------------------------------------- the decline


def test_a_decline_is_written_from_the_customers_own_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model may open on what happened before saying it is not ours."""

    monkeypatch.setattr(settings, "reply_rewrite_enabled", True)
    written = "يارب ما تشوف شر 🙂 — المعلومة دي مو عندي، كلّم خدمة العملاء."

    class Handler:
        def decline(self, text: str, language: str, timeout: float) -> str:
            assert "تأمين" in text  # the customer's turn, not a template
            return written

    monkeypatch.setattr("app.llm.get_llm_handler", lambda: Handler())
    assert templates.out_of_scope(Language.AR, turn="عملت حادثة وعايز تأمين") == written


@pytest.mark.parametrize(
    "candidate",
    [
        # A fee, a rate and a phone number are all facts we do not hold.
        "Transfers are free 🙂 — customer service can confirm; call 8001234567.",
        "It's 15 SAR 🙂 — customer service can help you with it.",
        # Forgot the one thing the customer can act on.
        "I don't have that information 🙂 — I can send money and pay bills.",
        # Answered in the wrong script.
        "لا أعرف — كلّم خدمة العملاء.",
    ],
)
def test_an_unsafe_decline_falls_back_to_the_fixed_wording(
    monkeypatch: pytest.MonkeyPatch, candidate: str
) -> None:
    monkeypatch.setattr(settings, "reply_rewrite_enabled", True)

    class Handler:
        def decline(self, text: str, language: str, timeout: float) -> str:
            return candidate

    monkeypatch.setattr("app.llm.get_llm_handler", lambda: Handler())
    assert (
        templates.out_of_scope(Language.EN, turn="what are the fees?")
        == (templates._OUT_OF_SCOPE[Language.EN][0])
    )


def test_a_decline_needs_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "reply_rewrite_enabled", True)
    monkeypatch.setattr("app.llm.get_llm_handler", lambda: None)
    assert (
        templates.out_of_scope(Language.AR, turn="عايز تأمين")
        == (templates._OUT_OF_SCOPE[Language.AR][0])
    )
