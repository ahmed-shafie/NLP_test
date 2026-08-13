"""Account-number and IBAN validation for the add-beneficiary flow.

Adding a beneficiary is a write, so the destination is validated properly here
rather than by "looks like it has digits": a full IBAN is checked against its
country length and the ISO 13616 mod-97 checksum, and a bare domestic account
number must be a plausible length.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

# IBAN total lengths for the markets we serve. Countries outside this table are
# accepted on structure + checksum alone.
_IBAN_LENGTHS: dict[str, int] = {
    "SA": 24,  # Saudi Arabia: SA + 2 check + 2 bank + 18 account
    "AE": 23,
    "EG": 29,
    "BH": 22,
    "KW": 30,
    "QA": 29,
    "JO": 30,
}

_IBAN_RE = re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{10,30}$")
_DIGITS_RE = re.compile(r"^\d+$")

# Domestic account numbers: shorter than 9 digits is a typo, not an account.
_MIN_ACCOUNT_DIGITS = 9
_MAX_ACCOUNT_DIGITS = 20


def normalize_account(raw: str) -> str:
    """Strip the spaces/dashes people paste IBANs with and upper-case it."""

    return re.sub(r"[\s\u00a0-]", "", raw).upper()


def looks_like_iban(value: str) -> bool:
    """True when the value is shaped like an IBAN (two letters, then digits)."""

    return bool(re.match(r"^[A-Z]{2}\d{2}", normalize_account(value)))


def iban_checksum_ok(value: str) -> bool:
    """ISO 13616 mod-97 check: rotate the first four chars, letters -> numbers."""

    account = normalize_account(value)
    rotated = account[4:] + account[:4]
    digits = "".join(
        str(ord(ch) - 55) if ch.isalpha() else ch  # 'A' -> 10 ... 'Z' -> 35
        for ch in rotated
    )
    if not _DIGITS_RE.match(digits):
        return False
    return int(digits) % 97 == 1


def validate_account(raw: str) -> tuple[str | None, str | None]:
    """Return ``(normalized_account, None)`` or ``(None, reason)``.

    ``reason`` is a stable machine key ("iban_length", "iban_checksum",
    "too_short", "not_an_account") that the templates turn into wording.
    """

    account = normalize_account(raw)
    if not account:
        return None, "not_an_account"

    if looks_like_iban(account):
        if not _IBAN_RE.match(account):
            return None, "not_an_account"
        expected = _IBAN_LENGTHS.get(account[:2])
        if expected is not None and len(account) != expected:
            return None, "iban_length"
        if not iban_checksum_ok(account):
            return None, "iban_checksum"
        return account, None

    if not _DIGITS_RE.match(account):
        return None, "not_an_account"
    if len(account) < _MIN_ACCOUNT_DIGITS:
        return None, "too_short"
    if len(account) > _MAX_ACCOUNT_DIGITS:
        return None, "not_an_account"
    return account, None


def expected_iban_length(country: str = "SA") -> int | None:
    return _IBAN_LENGTHS.get(country.upper())


@dataclass(frozen=True)
class IbanTypoHint:
    """Where a failed checksum says the typo probably is.

    The checksum proves *that* a character is wrong, never *which* one, so this
    only reports what the arithmetic actually pins down: ``swapped`` when one
    adjacent pair is the sole repair, ``positions`` (1-based) for the single
    characters that would each fix it on their own. Several positions means the
    location is genuinely unknown — the caller must not pick one.
    """

    swapped: tuple[int, int] | None = None
    positions: tuple[int, ...] = ()

    @property
    def is_located(self) -> bool:
        return self.swapped is not None or len(self.positions) == 1


def analyze_iban_typo(value: str) -> IbanTypoHint:
    """Search the single-character edits that would satisfy the mod-97 check."""

    account = normalize_account(value)
    if not _IBAN_RE.match(account) or iban_checksum_ok(account):
        return IbanTypoHint()

    swaps = [
        (index + 1, index + 2)
        for index in range(len(account) - 1)
        if account[index] != account[index + 1]
        and iban_checksum_ok(
            account[:index] + account[index + 1] + account[index] + account[index + 2 :]
        )
    ]

    positions = [
        index + 1
        for index in range(len(account))
        if any(
            char != account[index]
            and iban_checksum_ok(account[:index] + char + account[index + 1 :])
            for char in (
                string.digits if account[index].isdigit() else string.ascii_uppercase
            )
        )
    ]

    return IbanTypoHint(
        swapped=swaps[0] if len(swaps) == 1 else None,
        positions=tuple(positions),
    )
