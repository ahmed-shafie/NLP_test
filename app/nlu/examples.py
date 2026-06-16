"""Labeled example utterances used to seed the semantic intent index.

Each tuple is ``(utterance, intent)``. Examples cover English and Arabic
phrasings (including dialectal variants) for the money-transfer intent, plus
out-of-scope ``fallback`` examples so the classifier can reject unrelated input.
"""

from __future__ import annotations

from app.schemas import Intent

INTENT_EXAMPLES: list[tuple[str, Intent]] = [
    # --- transfer_money (English) ---
    ("transfer 500 dollars to John", Intent.TRANSFER_MONEY),
    ("send money to my friend Sara", Intent.TRANSFER_MONEY),
    ("I want to wire 1000 EUR to Ahmed", Intent.TRANSFER_MONEY),
    ("please move 250 from my savings to Lara", Intent.TRANSFER_MONEY),
    ("pay 75 pounds to the landlord", Intent.TRANSFER_MONEY),
    ("remit 300 dollars to my brother", Intent.TRANSFER_MONEY),
    ("can you send 50 to Mohamed", Intent.TRANSFER_MONEY),
    ("make a transfer of 2000 to account 12345", Intent.TRANSFER_MONEY),
    # --- transfer_money (Arabic) ---
    ("حوّل 500 جنيه إلى أحمد", Intent.TRANSFER_MONEY),
    ("ابعت فلوس لمحمد", Intent.TRANSFER_MONEY),
    ("ارسل 1000 ريال إلى سارة", Intent.TRANSFER_MONEY),
    ("عايز احول مبلغ لصديقي", Intent.TRANSFER_MONEY),
    ("حولي 300 درهم لعلي", Intent.TRANSFER_MONEY),
    ("ادفع 200 جنيه لصاحب البيت", Intent.TRANSFER_MONEY),
    ("اريد تحويل خمسمئة دولار الى خالد", Intent.TRANSFER_MONEY),
    ("حوّل من حسابي مبلغ إلى ليلى", Intent.TRANSFER_MONEY),
    # --- fallback / out of scope (English) ---
    ("what is my account balance", Intent.FALLBACK),
    ("show me my recent transactions", Intent.FALLBACK),
    ("how do I reset my password", Intent.FALLBACK),
    ("what is the weather today", Intent.FALLBACK),
    ("when does the branch open", Intent.FALLBACK),
    # --- fallback / out of scope (Arabic) ---
    ("ما هو رصيدي الحالي", Intent.FALLBACK),
    ("اعرض اخر العمليات", Intent.FALLBACK),
    ("كيف اغير كلمة المرور", Intent.FALLBACK),
    ("ما هي اسعار الفائدة", Intent.FALLBACK),
]
