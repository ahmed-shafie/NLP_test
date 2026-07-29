"""Labelled example utterances that seed the semantic intent index.

Mirrors ``app/nlu/examples.py``. Each tuple is ``(utterance, intent)``. Examples
cover English and Arabic phrasings so the multilingual embedder can match either
language. Out-of-scope ``FALLBACK`` examples teach the classifier to *reject*
unrelated input rather than forcing everything into an action.

# >>> EDIT PER CASE: when you add a new intent, add ~6-10 varied examples for it
#     here (both languages). More/《varied》 examples = better recall; this is how
#     you "teach" the classifier without any training or LLM.
"""

from __future__ import annotations

from service_template.schemas import Intent

INTENT_EXAMPLES: list[tuple[str, Intent]] = [
    # --- transfer_money (English) ---
    ("transfer 500 dollars to John", Intent.TRANSFER_MONEY),
    ("send money to my friend Sara", Intent.TRANSFER_MONEY),
    ("I want to wire 1000 EUR to Ahmed", Intent.TRANSFER_MONEY),
    ("please move 250 from my savings to Lara", Intent.TRANSFER_MONEY),
    ("pay 75 pounds to the landlord", Intent.TRANSFER_MONEY),
    ("remit 300 dollars to my brother", Intent.TRANSFER_MONEY),
    ("can you send 50 to Mohamed", Intent.TRANSFER_MONEY),
    ("I need to send some money", Intent.TRANSFER_MONEY),
    # --- transfer_money (Arabic) ---
    ("حوّل 500 ريال إلى أحمد", Intent.TRANSFER_MONEY),
    ("ابعت فلوس لمحمد", Intent.TRANSFER_MONEY),
    ("ارسل 1000 ريال إلى سارة", Intent.TRANSFER_MONEY),
    ("عايز احول مبلغ لصديقي", Intent.TRANSFER_MONEY),
    ("حولي 300 درهم لعلي", Intent.TRANSFER_MONEY),
    ("اريد تحويل خمسمئة دولار الى خالد", Intent.TRANSFER_MONEY),
    ("حوّل من حسابي مبلغ إلى ليلى", Intent.TRANSFER_MONEY),
    # --- small_talk / chit-chat (English) ---
    ("hi", Intent.SMALL_TALK),
    ("hello there", Intent.SMALL_TALK),
    ("hey, good morning", Intent.SMALL_TALK),
    ("thanks a lot", Intent.SMALL_TALK),
    ("thank you so much", Intent.SMALL_TALK),
    ("how are you doing", Intent.SMALL_TALK),
    ("goodbye", Intent.SMALL_TALK),
    # --- small_talk / chit-chat (Arabic) ---
    ("مرحبا", Intent.SMALL_TALK),
    ("اهلا وسهلا", Intent.SMALL_TALK),
    ("السلام عليكم", Intent.SMALL_TALK),
    ("شكرا جزيلا", Intent.SMALL_TALK),
    ("كيف حالك", Intent.SMALL_TALK),
    # --- fallback / out of scope (English) ---
    ("what is the weather today", Intent.FALLBACK),
    ("how do I reset my password", Intent.FALLBACK),
    ("when does the branch open", Intent.FALLBACK),
    ("show me my recent transactions", Intent.FALLBACK),
    # --- fallback / out of scope (Arabic) ---
    ("كيف اغير كلمة المرور", Intent.FALLBACK),
    ("ما هي اسعار الفائدة", Intent.FALLBACK),
    ("متى يفتح الفرع", Intent.FALLBACK),
]
