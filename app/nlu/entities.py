"""Language-agnostic slot extraction helpers (amount, currency, recipient).

These regex-based extractors work without any downloaded model and are used as a
baseline. The per-language modules (:mod:`app.nlu.english`, :mod:`app.nlu.arabic`)
augment recipient detection with spaCy/Stanza named-entity recognition.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from app.config import BILLER_CATEGORIES, CURRENCY_SYMBOLS, SUPPORTED_CURRENCIES
from app.data_loader import canonicalize_recipient, resolve_biller
from app.nlu.normalize import normalize
from app.schemas import BillEntities, Language

# Arabic-Indic (٠-٩) and Extended Arabic-Indic (۰-۹) digit translation to ASCII.
_DIGIT_MAP = {ord(c): str(i % 10) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹")}

# Multipliers for shorthand magnitudes (English + Arabic).
_MULTIPLIERS: dict[str, int] = {
    "k": 1_000,
    "thousand": 1_000,
    "m": 1_000_000,
    "million": 1_000_000,
    "الف": 1_000,
    "ألف": 1_000,
    "آلاف": 1_000,
    "مليون": 1_000_000,
}

_AMOUNT_RE = re.compile(
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    r"(?P<mult>k|m|thousand|million|الف|ألف|آلاف|مليون)?",
    re.IGNORECASE,
)

# "to John", "to my friend Sara"
_EN_RECIPIENT_RE = re.compile(
    r"\bto\s+(?:my\s+)?(?:friend\s+|account\s+|number\s+)?([A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*){0,2})",
)
# "إلى أحمد" / "الى احمد" / "لأحمد" (ل only when preceded by whitespace/start).
# "الي" is the common colloquial spelling of "إلى". The name also ends at a
# trailing amount, so "حوّل الى سارة 500" reads the recipient as "سارة".
_AR_RECIPIENT_RE = re.compile(
    r"(?:(?:إلى|الى|الي)\s+|(?:(?<=\s)|^)ل)"
    r"([^\d،,.]{2,40}?)(?:\s+(?:مبلغ|بمبلغ)|\s*[\d٠-٩]|$|[،,.])"
)

# "from my savings", "from current account", "from account ending 9988", ...
_EN_SOURCE_TYPE_RE = re.compile(
    r"\bfrom\s+(?:my\s+|the\s+)?"
    r"(savings|saving|current|checking|credit|salary)(?:\s+account)?\b",
    re.IGNORECASE,
)
_EN_SOURCE_ACCT_RE = re.compile(
    r"\bfrom\s+(?:my\s+|the\s+)?account\s+(?:ending\s+(?:in\s+)?|number\s+|no\.?\s*)?"
    r"(\w{2,})",
    re.IGNORECASE,
)
_EN_SOURCE_RE = re.compile(
    r"\bfrom\s+(?:my\s+)?([\w'’-]+(?:\s+[\w'’-]+){0,2}?)\s*account\b", re.IGNORECASE
)
_AR_SOURCE_RE = re.compile(r"من\s+حساب(?:ي)?\s*(?:ال)?([^\d،,.]{0,20})")

# Arabic words that are never a person, after the "إلى"/"ل" anchor. Without this
# "حوّل الى الايبان ..." names a payee called "الايبان" — and because the speller
# then corrects it to the real given name "أيبان", the account it belongs to is
# whoever that resolves to. Written unprefixed: ``_ar_word_stem`` strips ال/لل/ب.
_AR_NOT_A_NAME = frozenset(
    {
        "ايبان",
        "أيبان",
        "آيبان",
        "حساب",
        "حسابي",
        "حسابه",
        "الحساب",
        "رقم",
        "مبلغ",
        "فلوس",
        "فلوسي",
        "مصاري",
        "نقود",
        "كاش",
        "بطاقة",
        "بطاقتي",
        "محفظة",
        "محفظتي",
        "فاتورة",
        "فاتورتي",
        "فواتير",
        "سداد",
        "رصيد",
        "رصيدي",
        "شخص",
        "حد",
        "واحد",
        "شخصية",
        "المستفيد",
        "مستفيد",
        "خدمة",
        "العملاء",
        "عملاء",
        "بنك",
        "البنك",
        "فرع",
        "هذا",
        "هذه",
        "ذا",
        "دا",
        "هاد",
        "هادي",
        "نفس",
        "لا",
        "ولا",
        "نعم",
        "ايوه",
        "مش",
        "ماعرف",
    }
)
_AR_PREFIXES = ("لل", "ال", "بال", "و")

# --- lowercase English names -------------------------------------------------
# ``_EN_RECIPIENT_RE`` and spaCy both key on capitalisation, so a customer typing
# "send 50 to ahmed" loses the name entirely. The pass below reads it anyway, but
# only inside a payment-shaped sentence and only when every captured word could
# plausibly be a name — otherwise a question ("i need to talk to customer
# service") would invent a payee.

_EN_TO_RE = re.compile(r"\bto\b")
_EN_WORD_RE = re.compile(r"[a-z][a-z'’-]*")

# Skipped before the name starts: "to my brother ahmed".
_EN_NAME_LEAD = frozenset(
    {
        "my",
        "the",
        "a",
        "an",
        "his",
        "her",
        "their",
        "our",
        "friend",
        "brother",
        "sister",
        "colleague",
        "cousin",
        "wife",
        "husband",
    }
)

# Ends the name. Anything a bank customer says after "to" that is not a person.
_EN_NOT_A_NAME = frozenset(
    {
        # accounts, products, billers
        "savings",
        "saving",
        "current",
        "checking",
        "credit",
        "salary",
        "account",
        "accounts",
        "iban",
        "number",
        "no",
        "card",
        "wallet",
        "balance",
        "bill",
        "invoice",
        "biller",
        "electricity",
        "water",
        "internet",
        "phone",
        "mobile",
        "sadad",
        "stc",
        "mobily",
        "zain",
        # money words
        "money",
        "amount",
        "cash",
        "funds",
        "riyal",
        "riyals",
        "sr",
        "dollar",
        "dollars",
        "euro",
        "euros",
        "pound",
        "pounds",
        # verbs / function words
        "pay",
        "paying",
        "send",
        "sending",
        "transfer",
        "cover",
        "settle",
        "top",
        "confirm",
        "cancel",
        "make",
        "know",
        "talk",
        "add",
        "reset",
        "activate",
        "block",
        "check",
        "open",
        "close",
        "use",
        "get",
        "call",
        "contact",
        "complain",
        "report",
        "change",
        "from",
        "for",
        "with",
        "and",
        "please",
        "now",
        "today",
        "tomorrow",
        "it",
        "them",
        "him",
        "me",
        "you",
        "yes",
        "last",
        "next",
        # people-but-not-a-payee
        "someone",
        "somebody",
        "anyone",
        "person",
        "people",
        "guy",
        "man",
        "woman",
        "customer",
        "service",
        "support",
        "agent",
        "branch",
        "bank",
        "beneficiary",
        "recipient",
        "same",
        "other",
        "another",
        "wrong",
        "right",
        "correct",
        # question words
        "what",
        "which",
        "who",
        "where",
        "when",
        "why",
        "how",
        "this",
        "that",
        # "send 20 to my usual" is memory's job, not a payee called "usual"
        "usual",
        "favorite",
        "favourite",
        "regular",
        "default",
        "previous",
    }
    # Currency names/codes: "convert 500 sar to usd" names no payee.
    | {alias.lower() for aliases in SUPPORTED_CURRENCIES.values() for alias in aliases}
)

# "pay Mona 40 riyals": a payee named with no "to" anchor, recognised only when
# the very next thing is an amount and the sentence is not about a bill.
_EN_PAY_NAME_RE = re.compile(
    r"\b(?:pay|send|transfer|wire|remit|give)\s+"
    r"([A-Za-z][\w'\u2019-]*(?:\s+[A-Za-z][\w'\u2019-]*){0,2}?)\s+(?=\d)",
    re.IGNORECASE,
)

# A digit run this long is an account, an IBAN or a reference — not a price.
_ACCOUNT_DIGITS = 10
# ... or the words right before it say so.
_ACCOUNT_CUE_RE = re.compile(
    r"(?:\biban\b|\bacc(?:oun)?t\b|\ba/c\b|ايبان|أيبان|آيبان|حساب)" r"[^\d]{0,12}$",
    re.IGNORECASE,
)

_EN_SPELLED_AMOUNT = (
    "hundred",
    "thousand",
    "million",
    "fifty",
    "twenty",
    "ten",
    "five",
    "two",
    "one",
)
_EN_MOVE_VERBS = ("transfer", "send", "wire", "remit", "move", "pay")


def normalize_digits(text: str) -> str:
    """Convert Arabic-Indic digits to ASCII and normalise the decimal separator."""

    return text.translate(_DIGIT_MAP).replace("٫", ".").replace("٬", ",")


def extract_amount(text: str) -> Decimal | None:
    """Return the first monetary amount found, applying magnitude multipliers.

    A run of digits that identifies an account is skipped, never read as money:
    "حول الي ايبان ٠٦٦٦٣٥" was becoming a transfer of 66,635. An amount written
    in words ("حوّل ألف جنيه", "five hundred") is read here too, so no model is
    asked to supply the figure.
    """

    normalized = normalize_digits(text)
    for match in _AMOUNT_RE.finditer(normalized):
        if _identifies_an_account(normalized, match):
            continue
        raw = match.group("num").replace(",", "")
        try:
            value = Decimal(raw)
        except InvalidOperation:
            return None
        mult = match.group("mult")
        if mult:
            value *= _MULTIPLIERS[mult.lower()]
        return value
    return extract_spelled_amount(normalized)


# Amounts written in words. Composed left to right: a unit/tens/hundreds word
# adds, a magnitude word multiplies what came before it ("خمسة آلاف" = 5000).
_SPELLED_UNITS_RAW: dict[str, int] = {
    "واحد": 1,
    "واحده": 1,
    "اثنين": 2,
    "اتنين": 2,
    "ثلاثة": 3,
    "ثلاثه": 3,
    "تلاتة": 3,
    "تلاته": 3,
    "اربعة": 4,
    "اربعه": 4,
    "خمسة": 5,
    "خمسه": 5,
    "ستة": 6,
    "سته": 6,
    "سبعة": 7,
    "سبعه": 7,
    "ثمانية": 8,
    "تمانية": 8,
    "تسعة": 9,
    "تسعه": 9,
    "عشرة": 10,
    "عشره": 10,
    "عشرين": 20,
    "ثلاثين": 30,
    "تلاتين": 30,
    "اربعين": 40,
    "خمسين": 50,
    "ستين": 60,
    "سبعين": 70,
    "ثمانين": 80,
    "تمانين": 80,
    "تسعين": 90,
    "مية": 100,
    "ميه": 100,
    "مئة": 100,
    "ماية": 100,
    "ميتين": 200,
    "مئتين": 200,
    "ثلاثمية": 300,
    "ثلثمية": 300,
    "اربعمية": 400,
    "خمسمية": 500,
    "خمسمائة": 500,
    "ستمية": 600,
    "سبعمية": 700,
    "ثمانمية": 800,
    "تسعمية": 900,
    "الفين": 2_000,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SPELLED_MAGNITUDES_RAW: dict[str, int] = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "الف": 1_000,
    "آلاف": 1_000,
    "الاف": 1_000,
    "مليون": 1_000_000,
    "ملايين": 1_000_000,
}
# Looked up on normalized words, so the tables are normalized once here
# ("خمسمية" → "خمسميه", "ألف" → "الف").
_SPELLED_UNITS = {normalize(w): v for w, v in _SPELLED_UNITS_RAW.items()}
_SPELLED_MAGNITUDES = {normalize(w): v for w, v in _SPELLED_MAGNITUDES_RAW.items()}
_SPELLED_JOINERS = frozenset({"و", "and"})


def extract_spelled_amount(text: str) -> Decimal | None:
    """Read an amount written in words, or ``None`` when there isn't one."""

    total = 0
    current = 0
    seen = False
    magnified = False
    for raw_word in re.split(r"[\s,،.]+", normalize(text)):
        word = raw_word.strip()
        if not word or word in _SPELLED_JOINERS:
            continue
        unit = _SPELLED_UNITS.get(word)
        if unit is not None:
            current += unit
            seen = True
            continue
        magnitude = _SPELLED_MAGNITUDES.get(word)
        if magnitude is not None:
            total += (current or 1) * magnitude
            current = 0
            seen = magnified = True
            continue
        # Any other word ends the number; a later one starts afresh.
        total += current
        current = 0
    total += current
    if not seen or not total:
        return None
    # A bare small word ("one of my accounts") is only money when the sentence
    # also names a currency; a magnitude word ("ألف جنيه") speaks for itself.
    if not magnified and extract_currency(text) is None:
        return None
    return Decimal(total)


def _identifies_an_account(normalized: str, match: re.Match[str]) -> bool:
    """True when this run of digits is an account/IBAN rather than an amount.

    Either the customer said so ("iban", "حساب", "account number" right before it)
    or the run is too long to be money in this product.
    """

    digits = match.group("num").replace(",", "")
    if "." not in digits and len(digits) >= _ACCOUNT_DIGITS:
        return True
    return _ACCOUNT_CUE_RE.search(normalized[: match.start()]) is not None


def extract_currency(text: str) -> str | None:
    """Return the ISO-4217 code referenced in ``text``, if any."""

    for symbol, code in CURRENCY_SYMBOLS.items():
        if symbol in text:
            return code

    lowered = normalize_digits(text).lower()
    tokens = set(re.findall(r"[^\W\d_]+", lowered, re.UNICODE))
    for code, aliases in SUPPORTED_CURRENCIES.items():
        for alias in aliases:
            alias_l = alias.lower()
            if " " in alias_l:
                if alias_l in lowered:
                    return code
            elif alias_l in tokens:
                return code
    return None


def count_currencies(text: str) -> int:
    """Number of distinct ISO-4217 currencies referenced in ``text``.

    Two of them mean the message converts between them ("حوّل ٥٠٠ ريال لدولار")
    rather than moving money to somebody.
    """

    lowered = normalize_digits(text).lower()
    tokens = set(re.findall(r"[^\W\d_]+", lowered, re.UNICODE))
    # "لدولار" is "to dollars": Arabic proclitics glue onto the currency name.
    tokens |= {re.sub(r"^(?:ال|ول|بال|لل|ل|ب|و)", "", t) for t in tokens}
    found = {code for symbol, code in CURRENCY_SYMBOLS.items() if symbol in text}
    for code, aliases in SUPPORTED_CURRENCIES.items():
        for alias in aliases:
            alias_l = alias.lower()
            if (alias_l in lowered) if " " in alias_l else (alias_l in tokens):
                found.add(code)
                break
    return len(found)


def _has_payment_shape(lowered: str) -> bool:
    """True when the sentence asks to move money, not merely mentions banking."""

    if any(ch.isdigit() for ch in lowered):
        return True
    if any(word in lowered for word in _EN_SPELLED_AMOUNT):
        return True
    return any(verb in lowered for verb in _EN_MOVE_VERBS) and "money" in lowered


def extract_lowercase_recipient(text: str) -> str | None:
    """Read an all-lowercase English beneficiary name, or ``None``.

    The customer's spelling is returned verbatim: this path deliberately skips
    the gazetteer speller, so an unregistered name stays unregistered and the
    beneficiary lookup — not a fuzzy score — decides who gets paid.
    """

    lowered = normalize_digits(text).lower()
    if not _has_payment_shape(lowered):
        return None

    for anchor in _EN_TO_RE.finditer(lowered):
        words = _EN_WORD_RE.findall(lowered[anchor.end() :])
        name: list[str] = []
        for word in words[:5]:
            if not name and word in _EN_NAME_LEAD:
                continue
            if word == "to" or word in _EN_NOT_A_NAME:
                break
            name.append(word)
            if len(name) == 3:
                break
        if name:
            return " ".join(name)
    return None


def extract_recipient_without_anchor(text: str) -> str | None:
    """Read "pay Mona 40 riyals" — a payee named with no "to" before it.

    Without the "to" anchor the only evidence that the word is a person is its
    position: straight after a move verb and straight before an amount. That is
    also the shape of "pay mobily 100", so a message that resolves to a biller or
    mentions a bill is left alone — a biller must never become a payee.
    """

    if has_bill_word(text) or resolve_biller(text) is not None:
        return None
    match = _EN_PAY_NAME_RE.search(normalize_digits(text))
    if match is None:
        return None
    words = match.group(1).split()
    if any(w.lower() in _EN_NOT_A_NAME or w.lower() in _EN_NAME_LEAD for w in words):
        return None
    return " ".join(words)


def _ar_word_stem(word: str) -> str:
    """Strip the Arabic proclitics that glue onto a noun ("الايبان" → "ايبان")."""

    for prefix in _AR_PREFIXES:
        if word.startswith(prefix) and len(word) > len(prefix) + 1:
            return word[len(prefix) :]
    return word


def _is_ar_person_name(candidate: str) -> bool:
    """False when the captured words name an account, a bill or nobody at all.

    Checked before the speller runs: "الايبان" is one edit from the given name
    "أيبان", so a correction here would turn the word IBAN into a payee.
    """

    words = [w for w in re.split(r"\s+", candidate) if w]
    if not words:
        return False
    return not any(
        word in _AR_NOT_A_NAME or _ar_word_stem(word) in _AR_NOT_A_NAME
        for word in words
    )


def extract_recipient(text: str, language: Language) -> str | None:
    """Return the beneficiary name via language-specific surface patterns.

    Recognised given names are spelling-corrected against the name gazetteer
    (typos + Arabic/English transliteration); unknown names pass through as-is.
    """

    pattern = _AR_RECIPIENT_RE if language is Language.AR else _EN_RECIPIENT_RE
    match = pattern.search(text)
    if not match:
        if language is Language.AR:
            return None
        return extract_lowercase_recipient(text) or extract_recipient_without_anchor(
            text
        )
    candidate = match.group(1).strip(" ,،.")
    if not candidate:
        return None
    if language is Language.AR and not _is_ar_person_name(candidate):
        return None
    return canonicalize_recipient(candidate)


def extract_source_account(text: str, language: Language) -> str | None:
    """Return the source account hint (e.g. "savings"), if mentioned."""

    if language is Language.AR:
        match = _AR_SOURCE_RE.search(text)
        return match.group(1).strip(" ,،.") or None if match else None
    for pattern in (_EN_SOURCE_TYPE_RE, _EN_SOURCE_ACCT_RE, _EN_SOURCE_RE):
        match = pattern.search(text)
        if match:
            return match.group(1).strip(" ,،.") or None
    return None


# ---- Bill-payment slot extraction --------------------------------------- #

# A reference number following an explicit cue ("bill 778899", "ref 4455", "رقم ٩٩").
_REF_CUE_RE = re.compile(
    r"(?:\b(?:ref|reference|number|no|bill|account|invoice)\b|رقم|مرجع|فاتورة)"
    r"\s*[:#\-]?\s*(\d{2,})",
    re.IGNORECASE,
)
# An amount following an explicit cue ("amount 320", "for 210", "بمبلغ ٣٢٠").
_AMOUNT_CUE_RE = re.compile(
    r"(?:\bamount\b|\bfor\b|بمبلغ|مبلغ)\s*[:#]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE
)
_DIGITS_RUN_RE = re.compile(r"\d{2,}")
# A number written as money: a currency word or symbol against it on either side
# ("bill 100 sar", "ادفع فاتورة موبايلي ١٠٠ ريال", "SAR 100"). Priced like that,
# the number is the amount even where a reference would otherwise be read.
_CURRENCY_WORDS = "|".join(
    re.escape(alias)
    for alias in sorted(
        {a for aliases in SUPPORTED_CURRENCIES.values() for a in aliases}
        | set(CURRENCY_SYMBOLS),
        key=len,
        reverse=True,
    )
)
# Written whole, so "sara 100" is a name and a number, not 100 riyals.
_CURRENCY_TOKEN = rf"(?<![^\W\d_])(?:{_CURRENCY_WORDS})(?![^\W\d_])"
_MONEY_MARKED_RE = re.compile(
    rf"(?:{_CURRENCY_TOKEN}\s*(?P<before>\d+(?:\.\d+)?)"
    rf"|(?P<after>\d+(?:\.\d+)?)\s*{_CURRENCY_TOKEN})",
    re.IGNORECASE,
)
# A request to pay, as opposed to a bare answer naming a bill ("Water Services 5512").
_PAY_VERB_RE = re.compile(r"\b(?:pay|settle|cover)\b|ادفع|أدفع|سدد|دفع", re.IGNORECASE)
_BILL_WORD_RE = re.compile(r"\bbills?\b|\binvoices?\b|فاتورة|فواتير", re.IGNORECASE)
# The same word once :func:`normalize` has folded ة to ه ("فاتوره").
_BILL_WORD_FOLDED_RE = re.compile(
    r"\bbills?\b|\binvoices?\b|فاتوره|فواتير", re.IGNORECASE
)
# Free-text biller before the word "bill" (e.g. "City Power Co bill").
_EN_BILLER_RE = re.compile(
    r"([A-Za-z][\w&'’.-]*(?:\s+[A-Za-z][\w&'’.-]*){0,3})\s+bills?\b", re.IGNORECASE
)
# Free-text biller after "فاتورة" (e.g. "فاتورة شركة الكهرباء").
_AR_BILLER_RE = re.compile(r"فاتورة\s+([^\d،,.]{2,30})")
# Words that are never part of a biller name. Besides articles/possessives this
# covers the request preamble ("I want to pay a bill", "I need to settle a bill")
# so the verb phrase is not mistaken for the biller.
_BILLER_STOPWORDS = {
    "my",
    "the",
    "a",
    "an",
    "your",
    "our",
    "this",
    "pay",
    "paying",
    "settle",
    "settling",
    "i",
    "we",
    "you",
    "me",
    "us",
    "want",
    "wants",
    "wanna",
    "need",
    "needs",
    "would",
    "like",
    "to",
    "please",
    "let",
    "can",
    "could",
    "help",
}


def _strip_biller_stopwords(value: str) -> str:
    words = [w for w in value.split() if w.lower() not in _BILLER_STOPWORDS]
    return " ".join(words).strip(" ,،.")


def _to_decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw)
    except InvalidOperation:
        return None


def _amount_digits(amount: Decimal | None) -> str | None:
    if amount is None:
        return None
    return f"{amount.normalize():f}"


def _reference_via_cue(
    normalized: str, priced: frozenset[str] = frozenset()
) -> str | None:
    for match in _REF_CUE_RE.finditer(normalized):
        if match.group(1) not in priced:
            return match.group(1)
    return None


def _priced_runs(normalized: str) -> frozenset[str]:
    """Digit runs the customer priced with a currency, so they name money.

    "pay my mobily bill 100 sar" reads as the cue "bill" followed by 100, but
    the customer said what 100 is. A reference is never quoted in riyals.
    """

    return frozenset(
        run
        for match in _MONEY_MARKED_RE.finditer(normalized)
        for run in (match.group("before"), match.group("after"))
        if run
    )


def _bill_amount(
    normalized: str,
    currency: str | None,
    reference: str | None,
    *,
    biller_known: bool,
) -> Decimal | None:
    """Amount for a bill: an explicit cue, a currency-adjacent number, or —
    only when a biller is already known — a cue-less number placed where a
    reference cannot be (see :func:`_bare_number_amount`).

    A number that follows the bill word stays the reference, so "electricity
    bill 778899" doesn't read 778899 as the amount.
    """

    cue = _AMOUNT_CUE_RE.search(normalized)
    if cue is not None:
        return _to_decimal(cue.group(1))
    if currency:
        runs = re.findall(r"\d+(?:\.\d+)?", normalized)
        priced = _priced_runs(normalized)
        for run in runs:
            if run in priced:
                return _to_decimal(run)
        for run in runs:
            if reference is None or run != reference:
                return _to_decimal(run)
    if reference is None and biller_known:
        return _bare_number_amount(normalized)
    return None


def _bare_number_amount(normalized: str) -> Decimal | None:
    """A cue-less number in a bill message, when it can only be the amount.

    A reference follows what it refers to ("gas bill 5566", "فاتورة الغاز ٥٥٦٦"),
    so a number is read as money only when it comes *before* the bill word
    ("pay 320 for my phone bill") or when the message names a biller without
    saying "bill" at all ("pay mobily 100"). Both need the customer to be asking
    to pay: naming a bill in answer to "which bill?" ("Water Services 5512")
    gives the reference, not an amount.
    """

    run = _DIGITS_RUN_RE.search(normalized)
    if run is None or _PAY_VERB_RE.search(normalized) is None:
        return None
    bill_word = _BILL_WORD_RE.search(normalized)
    if bill_word is not None and run.start() > bill_word.start():
        return None
    return _to_decimal(run.group(0))


def extract_biller(
    text: str, language: Language, *, allow_semantic: bool = False
) -> tuple[str | None, str | None, str | None]:
    """Return ``(biller, category, biller_code)`` for a bill utterance.

    Resolution order: the SADAD catalogue (exact name/alias, then an optional
    FAISS fallback when ``allow_semantic`` is set) which yields a ``biller_code``;
    then the generic :data:`BILLER_CATEGORIES` keywords; then the free-text
    biller before "bill"/after "فاتورة".
    """

    record = resolve_biller(text, allow_semantic=allow_semantic)
    if record is not None:
        name = record.name_ar if language is Language.AR else record.name_en
        return name or record.name_en, record.category, record.biller_code

    lowered = normalize_digits(text).lower()
    for category, keywords in BILLER_CATEGORIES.items():
        if any(kw in lowered for kw in keywords):
            return category, category, None
    match = _EN_BILLER_RE.search(text) or _AR_BILLER_RE.search(text)
    if match:
        biller = _strip_biller_stopwords(match.group(1))
        return (biller or None), None, None
    return None, None, None


def extract_reference_number(text: str) -> str | None:
    """Return a bill reference: an explicit-cue number, else the first digit run."""

    normalized = normalize_digits(text)
    cue = _reference_via_cue(normalized)
    if cue is not None:
        return cue
    run = _DIGITS_RUN_RE.search(normalized)
    return run.group(0) if run else None


def has_bill_word(text: str) -> bool:
    """The word "bill" in either language, however the customer spelled it."""

    return bool(
        _BILL_WORD_RE.search(text) or _BILL_WORD_FOLDED_RE.search(normalize(text))
    )


def extract_bill_entities(
    text: str, language: Language, *, allow_semantic: bool = False
) -> BillEntities:
    """Extract all bill slots (biller, reference, amount, currency) from ``text``."""

    normalized = normalize_digits(text)
    biller, category, biller_code = extract_biller(
        text, language, allow_semantic=allow_semantic
    )
    currency = extract_currency(normalized)
    reference = _reference_via_cue(normalized, _priced_runs(normalized))
    amount = _bill_amount(
        normalized, currency, reference, biller_known=biller is not None
    )
    if reference is None and (biller is not None or has_bill_word(text)):
        amount_digits = _amount_digits(amount)
        for run in _DIGITS_RUN_RE.findall(normalized):
            if run != amount_digits:
                reference = run
                break
    return BillEntities(
        biller=biller,
        biller_category=category,
        biller_code=biller_code,
        biller_name=biller if biller_code else None,
        reference_number=reference,
        amount=amount,
        currency=currency,
    )
