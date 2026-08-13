"""Contextual answers for customer-service questions the assistant cannot execute.

The indexed corpus labels every refused row with the question's topic ("تم
التحصيل مرتين"), so a refused turn can be answered *about that topic* instead of
with the generic "transfer or bill?" menu. Each reply is hand-written and
reviewed: it names the topic, says plainly that this assistant cannot act on it,
offers only what the assistant really does (balance, beneficiaries, transfers,
bill payments) and points at human support.

Deliberately absent from every reply: durations, fee amounts, limits, and any
other bank policy. We do not have those facts, and a plausible-sounding invented
one is exactly the failure mode this two-tier design exists to prevent. Topics
are grouped into families so all 94 of them are covered; the most frequent ones
additionally carry a specific reply.

Neighbouring subjects are the main source of wrong answers, and the two remedies
here are deliberately different. Where the answer would be the same either way —
the exchange rate and its fee, a blocked card and a blocked PIN — the subjects
share one reply, so a mix-up cannot mislead anybody. Where the answers must
differ — "my card doesn't work" against "my card was stolen" — the question's own
words decide (:func:`_family_from_cues`), because retrieval demonstrably cannot.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from app.config import settings
from app.schemas import Language

# Every topic in the corpus maps to a family. Meta labels (``confirm``, ``deny``,
# ``out_of_scope``, ``repeat_request``) are intentionally absent: they are not
# questions about a subject, so a topical answer would be nonsense.
TOPIC_FAMILIES: dict[str, str] = {
    # Charged the wrong amount / twice.
    "تم التحصيل مرتين": "wrong_charge",
    "تلقي مبلغ خاطئ من النقد": "wrong_charge",
    "سعر الصرف الخاطئ للدفع بالبطاقة": "wrong_charge",
    "سعر صرف خاطئ للسحب النقدي": "wrong_charge",
    # Fees.
    "تحصيل رسوم التحويل": "fees",
    "تم تحصيل رسوم الدفع بالبطاقة": "fees",
    "رسوم السحب النقدي": "fees",
    "رسوم إضافية على كشف الحساب": "fees",
    "إعادة الشحن عن طريق رسوم التحويل المصرفي": "fees",
    # An existing transfer: late, failed, rejected, cancelled.
    "لم يستلم المستلم التحويل": "transfer_issue",
    "توقيت التحويل": "transfer_issue",
    "لم يتم تحديث الرصيد بعد التحويل المصرفي": "transfer_issue",
    "انتظار التحويل": "transfer_issue",
    "فشل التحويل": "transfer_issue",
    "رفض التحويل": "transfer_issue",
    "إلغاء التحويل": "transfer_issue",
    "نقل إلى الحساب": "transfer_issue",
    "المستفيد غير مسموح به": "transfer_issue",
    "transfer_status": "transfer_issue",
    "schedule_transfer": "transfer_issue",
    "cancel_operation": "transfer_issue",
    # Ordering, receiving and activating a card.
    "طلب بطاقة فعلية": "card_order",
    "وصول البطاقة": "card_order",
    "الحصول على بطاقة احتياطية": "card_order",
    "الحصول على بطاقة فعلية": "card_order",
    "الحصول على بطاقة افتراضية": "card_order",
    "الحصول على بطاقة افتراضية يمكن التخلص منها": "card_order",
    "تقدير تسليم البطاقة": "card_order",
    "بطاقة على وشك الانتهاء": "card_order",
    "تفعيل بطاقتي": "card_order",
    "ربط البطاقة": "card_order",
    # A card payment that failed, is pending, reversed or unrecognised.
    "إرجاع الدفع بالبطاقة؟": "card_payment_issue",
    "تم رفض الدفع بالبطاقة": "card_payment_issue",
    "الدفع بالبطاقة المعلق": "card_payment_issue",
    "الدفع بالبطاقة غير معترف به": "card_payment_issue",
    "قبول البطاقة": "card_payment_issue",
    "فيزا أو ماستر كارد": "card_payment_issue",
    "آبل باي أو جوجل باي": "card_payment_issue",
    "الدفع المباشر للدين غير معروف": "card_payment_issue",
    # ATM / cash.
    "رفض السحب النقدي": "cash_withdrawal",
    "في انتظار السحب النقدي": "cash_withdrawal",
    "السحب النقدي غير معترف به": "cash_withdrawal",
    "ابتلاع البطاقة": "cash_withdrawal",
    "دعم أجهزة الصراف الآلي": "cash_withdrawal",
    # Topping the balance up.
    "في انتظار التعبئة": "topup",
    "تعبئة تلقائية": "topup",
    "فشل التعبئة": "topup",
    "تعبئة الرصيد نقدًا أو بشيك": "topup",
    "عادت تعبئة الرصيد": "topup",
    "تحقق من تعبئة الرصيد": "topup",
    "حدود التعبئة": "topup",
    "التعبئة عن طريق البطاقة": "topup",
    "التعبئة عن طريق شحن البطاقة": "topup",
    "لم يتم تحديث الرصيد بعد الشيك أو الإيداع النقدي": "topup",
    # Fraud, theft, compromised card.
    "بطاقة مخترقة": "security",
    "الهاتف المفقود أو المسروق": "security",
    "البطاقة المفقودة أو المسروقة": "security",
    "report_fraud": "security",
    "freeze_account": "security",
    # A card that will not work: blocked card, blocked PIN, contactless dead.
    # The customer reports the same symptom for all three and the answer is the
    # same, so they share one reply rather than three that can be swapped.
    "البطاقة لا تعمل": "card_blocked",
    "البطاقة الافتراضية لا تعمل": "card_blocked",
    "عدم التلامس لا يعمل": "card_blocked",
    "رمز التعريف الشخصي محظور": "card_blocked",
    # PIN / passcode / sign-in.
    "نسيان رمز المرور": "pin_access",
    "تغيير رمز التعريف الشخصي": "pin_access",
    "change_pin": "pin_access",
    "logout": "pin_access",
    # Identity verification and source of funds.
    "لماذا التحقق من الهوية": "identity",
    "تحقق من هويتي": "identity",
    "غير قادر على التحقق من الهوية": "identity",
    "التحقق من مصدر الأموال": "identity",
    # Account administration.
    "تحرير التفاصيل الشخصية": "account_admin",
    "حد السن": "account_admin",
    "انهاء حساب": "account_admin",
    "دعم البلد": "account_admin",
    # Refunds and incoming money.
    "استرداد الأموال غير مرئي": "refunds",
    "طلب استرداد": "refunds",
    "تلقي الأموال": "refunds",
    # Exchanging currency: the rate and its fee are one subject to the customer,
    # and telling the two apart from the question alone is not reliable ("وين
    # ألقى سعر الصرف؟" is labelled رسوم الصرف), so one reply answers both.
    "سعر الصرف": "fx",
    "رسوم الصرف": "fx",
    "الصرف عبر التطبيق": "fx",
    "دعم العملات الورقية": "fx",
    "البطاقات والعملات المدعومة": "fx",
    "currency_conversion": "fx",
    # Limits.
    "حدود البطاقة التي يمكن التخلص منها": "limits",
    "set_limit": "limits",
    # Statements and history.
    "transaction_history": "records",
    "account_statement": "records",
    # Beneficiary administration we cannot perform (only add/list are supported).
    "delete_beneficiary": "beneficiaries_admin",
    "check_beneficiary": "beneficiaries_admin",
}

# One reviewed reply per family, so every mapped topic is covered.
FAMILY_REPLIES: dict[str, dict[Language, str]] = {
    "wrong_charge": {
        Language.AR: (
            "فاهم — سؤالك عن مبلغ اتحصّل بالغلط. أنا مساعد التحويلات "
            "والفواتير، فما أقدر أراجع عملية ولا أرجّع مبلغ؛ ده لازم يمشي مع "
            "خدمة العملاء. أقدر أوريك رصيدك دلوقتي لو يفيدك."
        ),
        Language.EN: (
            "I understand — this is about an amount charged incorrectly. I "
            "handle transfers and bill payments, so I can't review or reverse a "
            "charge; customer support has to do that. I can show you your "
            "balance if that helps."
        ),
    },
    "fees": {
        Language.AR: (
            "سؤالك عن رسوم. ما عندي جدول الرسوم ومش هخمّنه — خدمة العملاء "
            "تقدر تأكّدلك الرسم على عمليتك بالتحديد. أقدر أوريك رصيدك أو أساعدك "
            "في تحويل أو دفع فاتورة."
        ),
        Language.EN: (
            "This is a question about fees. I don't have the fee schedule and I "
            "won't guess it — customer support can confirm the exact fee on your "
            "transaction. I can show your balance, or help with a transfer or a "
            "bill payment."
        ),
    },
    "transfer_issue": {
        Language.AR: (
            "سؤالك عن تحويل قائم بالفعل. أنا أقدر أبدأ تحويل جديد، لكن "
            "متابعة أو تعديل أو إلغاء تحويل تم قبل كده مش من صلاحياتي — "
            "خدمة العملاء تتابعه لك. تحب أساعدك في تحويل جديد؟"
        ),
        Language.EN: (
            "This is about a transfer that already exists. I can start a new "
            "transfer, but tracking, changing or cancelling an earlier one is "
            "outside what I can do — customer support can follow it up. Would "
            "you like help with a new transfer?"
        ),
    },
    "card_order": {
        Language.AR: (
            "سؤالك عن إصدار بطاقة أو تفعيلها. ده بيتم من إدارة البطاقات وليس "
            "من هنا — خدمة العملاء تقدر تظبطه لك. أنا معاك في التحويلات ودفع "
            "الفواتير والرصيد."
        ),
        Language.EN: (
            "This is about issuing or activating a card. That's handled by card "
            "services, not here — customer support can sort it out. I'm here for "
            "transfers, bill payments and your balance."
        ),
    },
    "card_payment_issue": {
        Language.AR: (
            "سؤالك عن عملية دفع بالبطاقة. ما أقدر أراجع مدفوعات البطاقة ولا "
            "أعدّلها — خدمة العملاء هي اللي تتابع ده. أقدر أوريك رصيدك، أو "
            "أساعدك في تحويل أو دفع فاتورة."
        ),
        Language.EN: (
            "This is about a card payment. I can't review or change card "
            "payments — customer support handles those. I can show your balance, "
            "or help with a transfer or a bill payment."
        ),
    },
    "cash_withdrawal": {
        Language.AR: (
            "سؤالك عن سحب نقدي أو صراف آلي. عمليات الصراف مش من صلاحياتي، "
            "وخدمة العملاء تقدر تراجعها معاك. أقدر أوريك رصيدك."
        ),
        Language.EN: (
            "This is about a cash withdrawal or an ATM. ATM operations are "
            "outside what I can do, and customer support can review them with "
            "you. I can show your balance."
        ),
    },
    "topup": {
        Language.AR: (
            "سؤالك عن تعبئة رصيد أو إيداع. أنا ما أقدر أعبّي رصيد ولا أتابع "
            "إيداع — خدمة العملاء تقدر تتحقق منه. أقدر أوريك رصيدك الحالي."
        ),
        Language.EN: (
            "This is about a top-up or a deposit. I can't top up a balance or "
            "trace a deposit — customer support can check on it. I can show you "
            "your current balance."
        ),
    },
    "security": {
        Language.AR: (
            "لو بطاقتك أو جهازك ضاع أو تشك في عملية مش بتاعتك، كلّم خدمة "
            "العملاء فورًا لإيقاف البطاقة — ده مش شيء أقدر أعمله من هنا، "
            "والسرعة مهمة."
        ),
        Language.EN: (
            "If your card or phone was lost, or you suspect a transaction you "
            "didn't make, contact customer support right away to block the card "
            "— that's not something I can do from here, and speed matters."
        ),
    },
    "card_blocked": {
        Language.AR: (
            "فاهم — البطاقة مش شغالة أو محظورة (هي أو الرمز السري). ما أقدر "
            "أفكّ حظر بطاقة ولا أعيد تعيين رمز — ده بيتم مع خدمة العملاء أو من "
            "إعدادات البطاقة. لو البطاقة ضاعت أو تشك في عملية مش بتاعتك، "
            "كلّمهم فورًا. أنا معاك في التحويلات والفواتير والرصيد."
        ),
        Language.EN: (
            "Understood — the card won't work, or the card or PIN is blocked. I "
            "can't unblock a card or reset a PIN — that goes through customer "
            "support or your card settings. If the card was lost or you suspect "
            "a transaction you didn't make, contact them right away. I'm here "
            "for transfers, bills and your balance."
        ),
    },
    "pin_access": {
        Language.AR: (
            "سؤالك عن الرمز السري أو الدخول. ما أقدر أعرض ولا أغيّر رمزك — "
            "لأسباب أمنية ده بيتم مع خدمة العملاء أو من إعدادات البطاقة."
        ),
        Language.EN: (
            "This is about your PIN or sign-in. I can't show or change your PIN "
            "— for security that goes through customer support or your card "
            "settings."
        ),
    },
    "identity": {
        Language.AR: (
            "سؤالك عن التحقق من الهوية. أنا ما أتعامل مع مستندات التحقق ولا "
            "أعرف حالة ملفك — خدمة العملاء تقدر تقولك المطلوب بالتحديد."
        ),
        Language.EN: (
            "This is about identity verification. I don't handle verification "
            "documents and can't see your file's status — customer support can "
            "tell you exactly what's needed."
        ),
    },
    "account_admin": {
        Language.AR: (
            "سؤالك عن بيانات الحساب أو إجراء إداري عليه. التعديلات دي مش من "
            "صلاحياتي — خدمة العملاء تقدر تنفّذها. أنا معاك في التحويلات "
            "والفواتير والرصيد."
        ),
        Language.EN: (
            "This is about your account details or an administrative change. "
            "Those changes are outside what I can do — customer support can make "
            "them. I'm here for transfers, bills and your balance."
        ),
    },
    "refunds": {
        Language.AR: (
            "سؤالك عن مبلغ مسترد أو مبلغ داخل. أنا ما أقدر أتابع الاسترداد ولا "
            "أعرف مكانه — خدمة العملاء تقدر تراجعه. أقدر أوريك رصيدك الحالي."
        ),
        Language.EN: (
            "This is about a refund or incoming money. I can't trace a refund or "
            "see where it is — customer support can review it. I can show you "
            "your current balance."
        ),
    },
    "fx": {
        Language.AR: (
            "سؤالك عن تصريف العملة. أنا ما أعرض سعر الصرف ولا رسومه ومش "
            "هخمّنهم، وما أنفّذ تصريف — خدمة العملاء تقدر تأكّد السعر والرسم "
            "على عمليتك. بس أقدر أحوّل مبلغ بعملة محددة لمستفيد. تحب؟"
        ),
        Language.EN: (
            "This is about exchanging currency. I don't quote the rate or its "
            "fee and I won't guess either, and I can't exchange currency — "
            "customer support can confirm the rate and the fee on your "
            "transaction. I can send an amount in a given currency to a payee. "
            "Want to do that?"
        ),
    },
    "limits": {
        Language.AR: (
            "سؤالك عن الحدود. ما عندي حدود حسابك ومش هخمّنها — خدمة العملاء "
            "تقدر تأكّدها لك."
        ),
        Language.EN: (
            "This is about limits. I don't have your account's limits and I "
            "won't guess them — customer support can confirm them for you."
        ),
    },
    "records": {
        Language.AR: (
            "سؤالك عن كشف حساب أو سجل عمليات. أنا ما أعرض كشوف ولا سجل "
            "العمليات — أقدر أوريك رصيد حسابك، والكشف من خدمة العملاء أو "
            "التطبيق."
        ),
        Language.EN: (
            "This is about a statement or transaction history. I don't show "
            "statements or history — I can show your account balance, and a "
            "statement comes from customer support or the app."
        ),
    },
    "beneficiaries_admin": {
        Language.AR: (
            "سؤالك عن إدارة المستفيدين. أقدر أعرض مستفيديك أو أضيف مستفيد "
            "جديد، لكن الحذف أو التعديل مش من صلاحياتي. تحب أعرض القائمة؟"
        ),
        Language.EN: (
            "This is about managing payees. I can list your payees or add a new "
            "one, but deleting or editing them is outside what I can do. Want me "
            "to list them?"
        ),
    },
}

# Specific replies for the most frequent topics, which is where a family-level
# answer reads as evasive. Anything absent falls back to its family.
TOPIC_REPLIES: dict[str, dict[Language, str]] = {
    "تم التحصيل مرتين": {
        Language.AR: (
            "فاهم — المبلغ اتخصم مرتين. أنا ما أقدر أرجّع مبلغ ولا ألغي خصم، "
            "ده لازم يتفتح مع خدمة العملاء بتفاصيل العمليتين. أقدر أوريك "
            "رصيدك الحالي دلوقتي."
        ),
        Language.EN: (
            "I see — you were charged twice. I can't reverse or cancel a charge; "
            "that has to go to customer support with both transactions. I can "
            "show you your current balance right now."
        ),
    },
    "لم يستلم المستلم التحويل": {
        Language.AR: (
            "واضح — التحويل ما وصل للمستلم. أنا أقدر أبدأ تحويل جديد بس ما "
            "أقدر أتابع تحويل قديم ولا أعرف مكانه؛ خدمة العملاء تتابعه لك "
            "بمرجع العملية."
        ),
        Language.EN: (
            "Understood — the recipient hasn't received your transfer. I can "
            "start a new transfer but I can't trace an earlier one or see where "
            "it is; customer support can follow it up with the reference."
        ),
    },
    "إرجاع الدفع بالبطاقة؟": {
        Language.AR: (
            "سؤالك عن دفعة بالبطاقة رجعت. ما أقدر أعرف سبب الإرجاع ولا أعيد "
            "المحاولة — خدمة العملاء تقدر تشوف تفاصيل العملية. أقدر أوريك "
            "رصيدك أو أساعدك في تحويل."
        ),
        Language.EN: (
            "This is about a card payment that was reversed. I can't see why it "
            "was reversed or retry it — customer support can look at the "
            "transaction. I can show your balance or help with a transfer."
        ),
    },
    "توقيت التحويل": {
        Language.AR: (
            "سؤالك عن مدة وصول التحويل. ما عندي مدد التحويل ومش هخمّنها — "
            "خدمة العملاء تقدر تأكّدها حسب نوع التحويل. أقدر أساعدك في تحويل "
            "جديد."
        ),
        Language.EN: (
            "This is about how long a transfer takes. I don't have transfer "
            "timings and I won't guess them — customer support can confirm them "
            "for your transfer type. I can help you start a new transfer."
        ),
    },
    "تحصيل رسوم التحويل": {
        Language.AR: (
            "سؤالك عن رسم اتحصّل على تحويل. ما عندي جدول الرسوم ومش هخمّنه — "
            "خدمة العملاء تقدر تأكّد الرسم على تحويلك بالتحديد."
        ),
        Language.EN: (
            "This is about a fee charged on a transfer. I don't have the fee "
            "schedule and I won't guess it — customer support can confirm the "
            "fee on your specific transfer."
        ),
    },
    "لماذا التحقق من الهوية": {
        Language.AR: (
            "سؤالك ليه بنطلب تحقق من الهوية. أنا مش الجهة اللي تحدد متطلبات "
            "التحقق ولا أشوف ملفك — خدمة العملاء تقدر تشرح المطلوب في حالتك."
        ),
        Language.EN: (
            "You're asking why identity verification is needed. I don't set the "
            "verification requirements and can't see your file — customer "
            "support can explain what applies to you."
        ),
    },
    "المستفيد غير مسموح به": {
        Language.AR: (
            "سؤالك عن مستفيد مرفوض. أقدر أعرض مستفيديك أو أضيف واحد جديد، "
            "لكن سبب الرفض بيتحدد من النظام البنكي وخدمة العملاء توضّحه. تحب "
            "أعرض قائمة مستفيديك؟"
        ),
        Language.EN: (
            "This is about a payee that wasn't allowed. I can list your payees "
            "or add a new one, but the reason for the rejection comes from the "
            "banking system and support can explain it. Want me to list your "
            "payees?"
        ),
    },
    "رفض السحب النقدي": {
        Language.AR: (
            "سؤالك عن سحب نقدي مرفوض. ما أقدر أعرف سبب الرفض ولا أعيد "
            "المحاولة — خدمة العملاء تقدر تشوف العملية. أقدر أوريك رصيدك."
        ),
        Language.EN: (
            "This is about a declined cash withdrawal. I can't see why it was "
            "declined or retry it — customer support can look at the "
            "transaction. I can show you your balance."
        ),
    },
    "في انتظار التعبئة": {
        Language.AR: (
            "سؤالك عن تعبئة رصيد معلّقة. أنا ما أقدر أتابع التعبئة ولا "
            "أسرّعها — خدمة العملاء تقدر تتحقق من حالتها. أقدر أوريك رصيدك "
            "الحالي."
        ),
        Language.EN: (
            "This is about a pending top-up. I can't track or speed up a top-up "
            "— customer support can check its status. I can show you your "
            "current balance."
        ),
    },
    "استرداد الأموال غير مرئي": {
        Language.AR: (
            "سؤالك عن مبلغ مسترد ما ظهرش. ما أقدر أتابع الاسترداد ولا أعرف "
            "مكانه — خدمة العملاء تقدر تراجعه بمرجع العملية. أقدر أوريك "
            "رصيدك الحالي."
        ),
        Language.EN: (
            "This is about a refund that hasn't appeared. I can't trace a refund "
            "or see where it is — customer support can review it with the "
            "reference. I can show you your current balance."
        ),
    },
}


# Asking to stop the card now. Retrieval reads "freeze my card in the app" as a
# question about the app, and the answer to that one has to be the urgent one.
# "unblock"/"ألغي حظر" is the opposite request and must not match.
_FREEZE_CUE_RE = re.compile(
    r"\bfreeze\b|(?<!un)\bblock\b(?!ed)\s+(?:my\s+)?card"
    r"|جمد|تجميد|أجمد|إيقاف البطاقة|أوقف بطاق",
    re.IGNORECASE,
)
# A wallet question is its own subject, not a broken card.
_WALLET_RE = re.compile(r"apple\s*pay|google\s*pay|آبل باي|جوجل باي", re.IGNORECASE)
_CARD_WORD_RE = re.compile(r"\b(?:card|contactless)\b|بطاقت?|كارت", re.IGNORECASE)
# The symptom, in either order around the card word and across dialects.
_FAULT_CUE_RE = re.compile(
    r"not work|doesn'?t work|won'?t work|stopped work|can'?t use|cannot use"
    r"|\bblocked\b"
    r"|لا تعمل|لا يعمل|ما تعمل|ماتعمل|ما يعمل|ما تشتغل|ماتشتغل|مش شغال|معطل"
    r"|ما تخدم|ماغاديش تخدم|ما كتخدم|ماكتخدمش|متخدمش|ما خدامة|خدماش|خدامش"
    r"|ما نقدرش نستعمل|ما نقدرش نستخدم|ماني قادر استخدم|مش قادر أستخدم"
    r"|ما أقدر أستخدم|ما اقدر استخدم|محظور|محظورة",
    re.IGNORECASE,
)


def _family_from_cues(text: str, family: str) -> str:
    """Correct the retrieved family when the question's own words settle it.

    Retrieval confuses neighbouring card subjects — "freeze my card" lands on
    card ordering, "I can't use my card" lands on a stolen card — and the two
    answers are not interchangeable: one says "call support now to block it".
    So the urgent answer requires an urgent word in the question, and a card
    that merely does not work gets the blocked-card answer whatever was
    retrieved.
    """

    if _FREEZE_CUE_RE.search(text):
        return "security"
    if (
        _CARD_WORD_RE.search(text)
        and _FAULT_CUE_RE.search(text)
        and not _WALLET_RE.search(text)
    ):
        return "card_blocked"
    return family


def topic_reply(topic: str, language: Language) -> str | None:
    """Return the reviewed reply for ``topic``, or ``None`` if it has none.

    A specific reply wins over its family's; an unmapped topic (including the
    meta labels) returns ``None`` so the caller keeps the generic prompt.
    """

    specific = TOPIC_REPLIES.get(topic)
    if specific is not None:
        return specific[language]
    family = TOPIC_FAMILIES.get(topic)
    if family is None:
        return None
    return FAMILY_REPLIES[family][language]


@dataclass(frozen=True)
class TopicAnswer:
    """A reviewed answer and the retrieval that justified sending it."""

    reply: str
    subject: str
    score: float


def decide(
    text: str,
    top_score: float,
    votes: Mapping[str, int],
    retrieved: int,
    language: Language,
) -> TopicAnswer | None:
    """Choose an answer for the retrieved topics, or ``None`` to stay generic.

    Two levels of evidence, because answering about the wrong subject is the only
    real risk here:

    1. every retrieved row names the same topic — accepted at the lower
       ``topic_reply_unanimous_threshold``, since unanimity is what carries a
       cross-lingual query (an English question retrieves the right Arabic rows
       at ~0.81, below the bar a split vote has to clear);
    2. a majority names the same topic — accepted only at the full
       ``topic_reply_threshold``.

    A third level was measured and rejected: when the topics disagree but share a
    family, answering at family level roughly doubles coverage and *quadruples*
    wrong answers on the held-out slice (14-17% wrong vs 1.4%), because unrelated
    subjects share a family — "where is the card I ordered" and "my card was
    stolen" are not the same question. Disagreement now keeps the generic prompt.
    See ``research/vector_db_v08/topic_gate_sweep.py``.
    """

    if not votes:
        return None

    topic = max(votes, key=lambda t: votes[t])
    share = votes[topic] / retrieved if retrieved else 0.0
    bar = (
        settings.topic_reply_unanimous_threshold
        if share == 1.0
        else settings.topic_reply_threshold
    )
    if share < settings.topic_reply_agreement or top_score < bar:
        return None
    family = TOPIC_FAMILIES.get(topic)
    if family is not None:
        corrected = _family_from_cues(text, family)
        if corrected != family:
            return TopicAnswer(
                reply=FAMILY_REPLIES[corrected][language],
                subject=corrected,
                score=top_score,
            )
    reply = topic_reply(topic, language)
    if reply is None:
        return None
    return TopicAnswer(reply=reply, subject=topic, score=top_score)
