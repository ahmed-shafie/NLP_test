"""Calibrate the topic-answer gate for an English question against Banking77.

The gate in :mod:`app.conversation.topic_replies` was calibrated on the Arabic
held-out slice, because that is the only labelled slice we had — and the index
is 98.9% Arabic (31431 of 31781 rows). An English question therefore retrieves
Arabic rows, and cross-lingual similarity is systematically lower than the
same-language similarity the bar was set from: "my card is not working" retrieves
"البطاقة لا تعمل" nine times out of ten and still scores 0.9213, under the 0.94 a
split vote has to clear. The customer gets the generic menu for a question the
index answered correctly.

Measuring that needs English questions with gold topics. ArBanking77 is a
translation of Banking77, so the *English* Banking77 test split is exactly this
slice's counterpart: same 77 subjects, none of it indexed (one row overlaps and
is dropped). ``CATEGORY_TOPICS`` pairs each Banking77 category with the Arabic
topic string ArBanking77 translated it into.

Run with the Banking77 test CSV (not in the repo, Apache-2.0, from
PolyAI-LDN/task-specific-datasets)::

    BANKING77_TEST_CSV=/path/to/test.csv \\
        python -m research.vector_db_v08.topic_gate_english
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from app.conversation.topic_replies import (
    FAMILY_REPLIES,
    TOPIC_FAMILIES,
    _family_from_cues,
    topic_reply,
)
from app.nlu.corpus import load_corpus_examples
from app.nlu.semantic_intents import get_semantic_classifier
from app.schemas import Language

CSV_PATH = Path(os.environ.get("BANKING77_TEST_CSV", "data/banking77_test.csv"))
CACHE = Path(__file__).with_name("topic_evidence_en.jsonl")
CACHE_K = 15

# Banking77's category -> the Arabic topic string ArBanking77 translated it into.
CATEGORY_TOPICS: dict[str, str] = {
    "Refund_not_showing_up": "استرداد الأموال غير مرئي",
    "activate_my_card": "تفعيل بطاقتي",
    "age_limit": "حد السن",
    "apple_pay_or_google_pay": "آبل باي أو جوجل باي",
    "atm_support": "دعم أجهزة الصراف الآلي",
    "automatic_top_up": "تعبئة تلقائية",
    "balance_not_updated_after_bank_transfer": (
        "لم يتم تحديث الرصيد بعد التحويل المصرفي"
    ),
    "balance_not_updated_after_cheque_or_cash_deposit": (
        "لم يتم تحديث الرصيد بعد الشيك أو الإيداع النقدي"
    ),
    "beneficiary_not_allowed": "المستفيد غير مسموح به",
    "cancel_transfer": "إلغاء التحويل",
    "card_about_to_expire": "بطاقة على وشك الانتهاء",
    "card_acceptance": "قبول البطاقة",
    "card_arrival": "وصول البطاقة",
    "card_delivery_estimate": "تقدير تسليم البطاقة",
    "card_linking": "ربط البطاقة",
    "card_not_working": "البطاقة لا تعمل",
    "card_payment_fee_charged": "تم تحصيل رسوم الدفع بالبطاقة",
    "card_payment_not_recognised": "الدفع بالبطاقة غير معترف به",
    "card_payment_wrong_exchange_rate": "سعر الصرف الخاطئ للدفع بالبطاقة",
    "card_swallowed": "ابتلاع البطاقة",
    "cash_withdrawal_charge": "رسوم السحب النقدي",
    "cash_withdrawal_not_recognised": "السحب النقدي غير معترف به",
    "change_pin": "تغيير رمز التعريف الشخصي",
    "compromised_card": "بطاقة مخترقة",
    "contactless_not_working": "عدم التلامس لا يعمل",
    "country_support": "دعم البلد",
    "declined_card_payment": "تم رفض الدفع بالبطاقة",
    "declined_cash_withdrawal": "رفض السحب النقدي",
    "declined_transfer": "رفض التحويل",
    "direct_debit_payment_not_recognised": "الدفع المباشر للدين غير معروف",
    "disposable_card_limits": "حدود البطاقة التي يمكن التخلص منها",
    "edit_personal_details": "تحرير التفاصيل الشخصية",
    "exchange_charge": "رسوم الصرف",
    "exchange_rate": "سعر الصرف",
    "exchange_via_app": "الصرف عبر التطبيق",
    "extra_charge_on_statement": "رسوم إضافية على كشف الحساب",
    "failed_transfer": "فشل التحويل",
    "fiat_currency_support": "دعم العملات الورقية",
    "get_disposable_virtual_card": "الحصول على بطاقة افتراضية يمكن التخلص منها",
    "get_physical_card": "الحصول على بطاقة فعلية",
    "getting_spare_card": "الحصول على بطاقة احتياطية",
    "getting_virtual_card": "الحصول على بطاقة افتراضية",
    "lost_or_stolen_card": "البطاقة المفقودة أو المسروقة",
    "lost_or_stolen_phone": "الهاتف المفقود أو المسروق",
    "order_physical_card": "طلب بطاقة فعلية",
    "passcode_forgotten": "نسيان رمز المرور",
    "pending_card_payment": "الدفع بالبطاقة المعلق",
    "pending_cash_withdrawal": "في انتظار السحب النقدي",
    "pending_top_up": "في انتظار التعبئة",
    "pending_transfer": "انتظار التحويل",
    "pin_blocked": "رمز التعريف الشخصي محظور",
    "receiving_money": "تلقي الأموال",
    "request_refund": "طلب استرداد",
    "reverted_card_payment?": "إرجاع الدفع بالبطاقة؟",
    "supported_cards_and_currencies": "البطاقات والعملات المدعومة",
    "terminate_account": "انهاء حساب",
    "top_up_by_bank_transfer_charge": "إعادة الشحن عن طريق رسوم التحويل المصرفي",
    "top_up_by_card_charge": "التعبئة عن طريق شحن البطاقة",
    "top_up_by_cash_or_cheque": "تعبئة الرصيد نقدًا أو بشيك",
    "top_up_failed": "فشل التعبئة",
    "top_up_limits": "حدود التعبئة",
    "top_up_reverted": "عادت تعبئة الرصيد",
    "topping_up_by_card": "التعبئة عن طريق البطاقة",
    "transaction_charged_twice": "تم التحصيل مرتين",
    "transfer_fee_charged": "تحصيل رسوم التحويل",
    "transfer_into_account": "نقل إلى الحساب",
    "transfer_not_received_by_recipient": "لم يستلم المستلم التحويل",
    "transfer_timing": "توقيت التحويل",
    "unable_to_verify_identity": "غير قادر على التحقق من الهوية",
    "verify_my_identity": "تحقق من هويتي",
    "verify_source_of_funds": "التحقق من مصدر الأموال",
    "verify_top_up": "تحقق من تعبئة الرصيد",
    "virtual_card_not_working": "البطاقة الافتراضية لا تعمل",
    "visa_or_mastercard": "فيزا أو ماستر كارد",
    "why_verify_identity": "لماذا التحقق من الهوية",
    "wrong_amount_of_cash_received": "تلقي مبلغ خاطئ من النقد",
    "wrong_exchange_rate_for_cash_withdrawal": "سعر صرف خاطئ للسحب النقدي",
}


@dataclass(frozen=True)
class Cached:
    """An English question, its gold Arabic topic, and its cached neighbours."""

    text: str
    gold: str
    neighbours: tuple[tuple[str, float], ...]

    def evidence(self, k: int) -> tuple[float, dict[str, int], int]:
        window = self.neighbours[:k]
        votes: dict[str, int] = defaultdict(int)
        for topic, _ in window:
            if topic:
                votes[topic] += 1
        top = window[0][1] if window else 0.0
        return top, dict(votes), len(window)


def build_cache() -> None:
    classifier = get_semantic_classifier()
    if classifier is None:
        raise SystemExit("semantic classifier unavailable")
    indexed = {row.text.strip().lower() for row in load_corpus_examples()}
    with (
        CSV_PATH.open(encoding="utf-8") as source,
        CACHE.open("w", encoding="utf-8") as handle,
    ):
        for done, row in enumerate(csv.DictReader(source)):
            text = row["text"].strip()
            if text.lower() in indexed:  # never score retrieval of a memorised row
                continue
            if done % 500 == 0:
                print(f"  {done}", flush=True)
            neighbours = classifier.similar(text, k=CACHE_K)
            handle.write(
                json.dumps(
                    {
                        "text": text,
                        "gold": CATEGORY_TOPICS[row["category"]],
                        "neighbours": [
                            [n.topic, round(n.score, 4)] for n in neighbours
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def read_cache() -> Iterator[Cached]:
    with CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            yield Cached(
                text=raw["text"],
                gold=raw["gold"],
                neighbours=tuple((t, s) for t, s in raw["neighbours"]),
            )


def score(
    rows: list[Cached], k: int, unanimous: float, majority: float, agreement: float
) -> tuple[int, int, list[tuple[str, str, str]]]:
    """Return (answered, wrong text, wrong examples) for one candidate gate."""

    answered = 0
    wrong = 0
    examples: list[tuple[str, str, str]] = []
    for row in rows:
        top, votes, retrieved = row.evidence(k)
        if not votes:
            continue
        topic = max(votes, key=lambda t: votes[t])
        share = votes[topic] / retrieved if retrieved else 0.0
        bar = unanimous if share == 1.0 else majority
        if share < agreement or top < bar:
            continue
        family = TOPIC_FAMILIES.get(topic)
        subject = topic
        reply: str | None
        if family is not None:
            corrected = _family_from_cues(row.text, family)
            if corrected != family:
                reply = FAMILY_REPLIES[corrected][Language.EN]
                subject = corrected
            else:
                reply = topic_reply(topic, Language.EN)
        else:
            reply = topic_reply(topic, Language.EN)
        if reply is None:
            continue
        answered += 1
        if reply != topic_reply(row.gold, Language.EN):
            wrong += 1
            if len(examples) < 12:
                examples.append((row.gold, subject, row.text))
    return answered, wrong, examples


def main() -> None:
    if not CACHE.exists():
        build_cache()
    rows = list(read_cache())
    total = len(rows)
    print(f"English held-out rows: {total}\n")
    print(
        f"{'k':>3} {'unanimous':>9} {'majority':>8} {'agree':>5} "
        f"{'answered':>18} {'wrong':>16}"
    )
    for k in (5, 7, 10, 15):
        for uni in (0.66, 0.70, 0.74, 0.78):
            for maj in (0.78, 0.82, 0.86, 0.90, 0.94):
                for agree in (0.6, 0.8):
                    answered, wrong, _ = score(rows, k, uni, maj, agree)
                    if not answered:
                        continue
                    share = f"{answered}/{total} = {answered / total:5.1%}"
                    bad = f"{wrong}/{answered} = {wrong / answered:5.1%}"
                    print(
                        f"{k:3d} {uni:9.2f} {maj:8.2f} {agree:5.1f} "
                        f"{share:>18} {bad:>16}"
                    )


if __name__ == "__main__":
    main()
