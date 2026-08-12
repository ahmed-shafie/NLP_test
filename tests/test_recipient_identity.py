"""A recipient is never silently swapped for a similar-sounding other person.

Two independent places used to change who a transfer names: the name gazetteer
typo-corrected "Noura" into the different name "Nouran", and contact matching
accepted the nearest embedding neighbour, which for "محمد نور" is the record of
محمد علي. Both must now decline and leave the question to the customer.
"""

from app.data_loader import canonicalize_recipient, lookup_name
from app.nlu.contacts import ContactMatcher
from app.schemas import Contact

# Two people who share a first name, plus one listed in both scripts.
CONTACTS = [
    Contact(id="1", name="Mohamed Ali", account="EG1002"),
    Contact(id="2", name="محمد علي", account="EG1002"),
    Contact(id="3", name="محمد نور", account="EG1009"),
    Contact(id="4", name="Laila Mansour", account="EG1004"),
]


def test_a_name_is_not_corrected_into_a_different_name() -> None:
    assert lookup_name("noura") is None
    assert canonicalize_recipient("Noura Saad") == "Noura Saad"


def test_a_tie_between_two_names_is_not_broken_silently() -> None:
    # "ahmd" is one edit from both Ahmad and Ahmed.
    assert lookup_name("ahmd") is None


def test_real_typos_are_still_corrected() -> None:
    assert lookup_name("mohamd") == "Mohamed"
    assert lookup_name("abdulah") == "Abdullah"
    assert canonicalize_recipient("احمد حسن") == "أحمد حسن"


def test_a_look_alike_contact_is_not_offered() -> None:
    matcher = ContactMatcher([CONTACTS[0], CONTACTS[1], CONTACTS[3]])
    matched, candidates = matcher.resolve("محمد نور")
    assert matched is None
    # The ranking is still reported: it is evidence, not a decision.
    assert candidates


def test_one_person_listed_in_both_scripts_still_resolves() -> None:
    matcher = ContactMatcher([CONTACTS[0], CONTACTS[1], CONTACTS[3]])
    matched, _ = matcher.resolve("محمد علي")
    assert matched is not None
    assert matched.contact.account == "EG1002"


def test_two_people_sharing_a_first_name_are_not_guessed() -> None:
    matcher = ContactMatcher(CONTACTS)
    matched, _ = matcher.resolve("محمد")
    assert matched is None


def test_an_unrelated_name_matches_nobody() -> None:
    matcher = ContactMatcher(CONTACTS)
    matched, _ = matcher.resolve("نورة سعد")
    assert matched is None
