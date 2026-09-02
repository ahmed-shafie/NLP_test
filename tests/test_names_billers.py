"""Tests for SADAD biller resolution, the name gazetteer, SAR default, chit-chat."""

from __future__ import annotations

import pytest

from app.conversation import templates
from app.conversation.engine import ConversationEngine
from app.conversation.state import ConversationStatus
from app.data_loader import (
    canonicalize_recipient,
    is_known_name,
    lookup_name,
    resolve_biller,
    resolve_biller_by_code,
    resolve_biller_candidates,
    resolve_biller_fuzzy,
    resolve_biller_gazetteer,
    transliterations,
)
from app.nlu.normalize import normalize, normalize_tokens
from app.schemas import Intent, Language


@pytest.fixture()
def engine() -> ConversationEngine:
    return ConversationEngine()


# ------------------------------ normalization ----------------------------- #


def test_normalize_unifies_arabic_forms():
    # Diacritics + alef/ya variants collapse to a single matching key.
    assert normalize("أَحْمَد") == normalize("احمد")
    # alef-madda + ta-marbuta normalise to bare alef + ha.
    assert normalize("آمنة") == normalize("امنه")


def test_normalize_collapses_repeats_and_case():
    assert normalize("YESSS") == "yes"
    assert normalize_tokens("  Hello   World ") == ["hello", "world"]


# --------------------------- biller resolution ---------------------------- #


def test_biller_gazetteer_exact_name():
    rec = resolve_biller_gazetteer("pay my STC bill 778899")
    assert rec is not None
    assert rec.biller_code == "001"


def test_biller_generic_alias_maps_to_sadad_code():
    assert resolve_biller_gazetteer("electricity").biller_code == "002"
    assert resolve_biller_gazetteer("ادفع فاتورة الكهرباء").biller_code == "002"
    assert resolve_biller_gazetteer("water").biller_code == "015"


@pytest.mark.parametrize("term", ["مياه", "المياه", "مية", "موية", "الموية", "مويه"])
def test_biller_colloquial_water_spellings(term: str):
    # "موية"/"مويه" is how customers say "مياه" in the Gulf dialect.
    assert resolve_biller_gazetteer(f"ادفع فاتورة {term}").biller_code == "015"


@pytest.mark.parametrize(
    "text",
    [
        "ادفع فاتورة اس تي سي",  # brand spelled out letter-by-letter in Arabic
        "ادفع ل اس تي سي",
        "ادفع لاس تي سي",  # "ل" fused onto the first token
        "فاتورة إس تي سي",
    ],
)
def test_biller_arabic_letter_spelling_of_stc(text: str):
    assert resolve_biller_gazetteer(text).biller_code == "001"


def test_biller_matches_through_fused_arabic_prefix():
    # "للمياه" is "لـ" + "المياه"; "وزين" is "و" + "زين".
    assert resolve_biller_gazetteer("ادفع فاتورة وزين").biller_code == "044"
    assert resolve_biller_gazetteer("فاتورة للكهرباء").biller_code == "002"


@pytest.mark.parametrize(
    "text", ["ادفع الايجار", "ادفع الإيجار", "ابغى اسدد الايجار حقي"]
)
def test_biller_matches_through_the_definite_article(text: str):
    """The catalogue lists "إيجار"; customers type "الايجار"."""

    rec = resolve_biller_gazetteer(text)
    assert rec is not None
    assert rec.biller_code == "153"


def test_biller_semantic_disabled_by_default():
    # Arbitrary chit-chat must not be mis-resolved to a near-neighbour biller.
    assert resolve_biller_gazetteer("hello") is None
    assert resolve_biller("hello") is None  # allow_semantic defaults to False


# ----------------------- biller candidates (C2) --------------------------- #


def test_ambiguous_generic_term_returns_several_candidates():
    cands = resolve_biller_candidates("pay my electricity bill")
    codes = [rec.biller_code for rec in cands]
    assert codes == ["002", "004"]  # Saudi Electric Company then Marafiq


def test_named_biller_returns_single_candidate():
    cands = resolve_biller_candidates("pay my STC bill 778899")
    assert [rec.biller_code for rec in cands] == ["001"]


def test_no_match_returns_empty_candidates():
    assert resolve_biller_candidates("hello there") == []


def test_generic_category_term_lists_all_billers_in_category():
    # "internet" must offer every SADAD Telecom & Internet biller (STC, Mobily,
    # Zain, ...), not just names that literally contain the word "internet".
    cands = resolve_biller_candidates("pay my internet bill")
    codes = {rec.biller_code for rec in cands}
    assert len(cands) > 5
    assert "001" in codes  # STC
    assert "005" in codes  # Mobily
    assert all(rec.category == "Telecom & Internet" for rec in cands)


def test_freetext_multiword_name_is_not_swallowed_by_category():
    # A specific (unknown) biller name must stay free text, not collapse to a
    # whole category.
    assert resolve_biller_candidates("pay my Acme Telecom bill") == []


def test_numeric_token_resolves_to_sadad_code():
    assert resolve_biller_by_code("153").name_en == "Ejar"
    assert resolve_biller_by_code("001").biller_code == "001"
    assert resolve_biller_by_code("1").biller_code == "001"  # zero-padded
    assert resolve_biller_by_code("٠٠١").biller_code == "001"  # Arabic-Indic
    assert resolve_biller_by_code("778899") is None  # too long -> a reference
    assert resolve_biller_by_code("999") is None  # not a real code


def test_a_misspelt_utility_word_still_reaches_the_utility():
    """ "قهرباء"/"كرباء" is "كهرباء": one slipped letter must not lose the bill."""

    for typo in ("قهرباء", "كرباء", "ادفع فتوره قهرباء", "electricty"):
        codes = [
            rec.biller_code
            for rec in resolve_biller_candidates(typo, allow_semantic=True)
        ]
        assert codes == ["002", "004"], typo


def test_a_misspelt_generic_word_keeps_asking_which_biller():
    cands = resolve_biller_candidates("قهرباء", allow_semantic=True)
    assert len(cands) > 1


def test_a_bill_verb_alone_resolves_nothing():
    assert resolve_biller_candidates("ادفع", allow_semantic=True) == []
    assert resolve_biller_candidates("فاتورة", allow_semantic=True) == []


def test_fuzzy_typo_matches_biller_name():
    assert resolve_biller_fuzzy("egar").name_en == "Ejar"  # single-letter typo
    assert resolve_biller_fuzzy("mobiley").name_en == "Mobily"


def test_fuzzy_does_not_match_common_shared_words():
    # A word shared by many billers must not fuzzy-resolve to an arbitrary one.
    assert resolve_biller_fuzzy("saudi") is None


# --------------------- cross-script transliteration (C1) ------------------ #


def test_transliterations_bridge_scripts():
    assert "محمد" in transliterations("mohammed")
    assert "mohammed" in transliterations("محمد")
    assert transliterations("Ahmed") == {normalize("أحمد")}


def test_transliterations_unknown_token_is_empty():
    assert transliterations("zzzqx") == set()


# ----------------------------- name gazetteer ----------------------------- #


def test_lookup_name_exact_and_typo():
    assert lookup_name("Ahmed") == "Ahmed"
    # A typo fuzzy-matches to a recognised given name, as long as only one name
    # is close to it ("Ahmd" is equally close to Ahmad and Ahmed, and correcting
    # it either way would name a different person - see
    # tests/test_recipient_identity.py).
    assert lookup_name("Mohamd") == "Mohamed"


def test_lookup_name_preserves_script():
    # An Arabic name stays Arabic (no silent transliteration to English).
    resolved = lookup_name("احمد")
    assert resolved is not None
    assert any("\u0600" <= ch <= "\u06ff" for ch in resolved)


def test_canonicalize_keeps_unknown_tokens():
    assert is_known_name("Ahmed") is True
    # Known token corrected, unknown surname preserved as-is.
    assert canonicalize_recipient("Ahmd Zzzqx").endswith("Zzzqx")


# ------------------------------ SAR default ------------------------------- #


def test_transfer_defaults_to_sar(engine: ConversationEngine):
    result = engine.handle("send 50 to Ahmed", "sar1")
    assert result.state.intent is Intent.TRANSFER_MONEY
    assert result.state.slots.currency == "SAR"


def test_bill_defaults_to_sar(engine: ConversationEngine):
    result = engine.handle("pay electricity bill 778899 amount 320", "sar2")
    assert result.state.intent is Intent.PAY_BILL
    assert result.state.slots.currency == "SAR"


# ------------------------------- chit-chat -------------------------------- #


def test_is_small_talk_only_for_pure_chitchat():
    assert templates.is_small_talk("hi") is True
    assert templates.is_small_talk("thanks") is True
    assert templates.is_small_talk("how are you") is True
    assert templates.is_small_talk("hi I want to pay a bill") is False
    assert templates.is_small_talk("pay my electricity bill") is False


def test_small_talk_replies_are_warm_and_bilingual():
    en = templates.small_talk("hi", Language.EN)
    ar = templates.small_talk("شكرا", Language.AR)
    assert en and ar
    # Arabic reply actually contains Arabic script.
    assert any("\u0600" <= ch <= "\u06ff" for ch in ar)


def test_engine_greets_then_awaits_choice(engine: ConversationEngine):
    greeting = engine.handle("hi", "cc1")
    assert greeting.state.status is ConversationStatus.SELECTING
    # A bare menu choice after the greeting still routes correctly.
    chosen = engine.handle("2", "cc1")
    assert chosen.state.intent is Intent.PAY_BILL
