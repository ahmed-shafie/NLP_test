# Design — Add "Pay Bills" alongside Transfers

Status: **proposal (nothing implemented yet)** · App role: pure NLP/conversation
middleware (text → structured attributes for the payment hub).

---

## 1. Why a new intent (not "a transfer to a biller")

Today the pipeline assumes a single intent `TRANSFER_MONEY` with required slots
`amount / currency / recipient`. A bill payment has a different shape — you pay a
**biller** against a **reference number**, not a person. Forcing it into the
transfer model would overload `recipient` and lose the reference number. So we add
a first-class intent `PAY_BILL` with its own slots and its own structured payload.

```
Intent: TRANSFER_MONEY | PAY_BILL | FALLBACK     # PAY_BILL is new
```

---

## 2. Slots

| Slot | Transfer | Pay bill | Notes |
|---|---|---|---|
| `amount` | required | required | customer states it (hub owns bill data) |
| `currency` | required | required | habit/default can fill (e.g. EGP) |
| `recipient` | required | — | person/beneficiary |
| `biller` | — | **required** | e.g. electricity, water, Vodafone |
| `reference_number` | — | **required** | bill / subscriber / account number |
| `note` | optional | optional | free text |

Per-intent required sets:
- `TRANSFER_MONEY` → `amount, currency, recipient`
- `PAY_BILL` → `biller, reference_number, amount, currency`

`biller` resolves against a **configurable category list** with a free-text
fallback:
```
biller_categories = {
  "electricity": ["electricity", "power", "كهرباء"],
  "water":       ["water", "مياه", "مية"],
  "gas":         ["gas", "غاز"],
  "internet":    ["internet", "wifi", "نت", "انترنت"],
  "mobile":      ["mobile", "phone", "vodafone", "orange", "etisalat", "we",
                  "موبايل", "فودافون", "اورنج"],
  "tv":          ["tv", "cable", "تليفزيون"],
}
```
Anything not matched is kept as the raw biller string (e.g. "City Power Co").

---

## 3. Conversation flows (concrete)

### 3.1 English — all slots in one turn
```
U: pay 320 EGP electricity bill 778899
B: Please confirm: pay 320 EGP to electricity (ref 778899). Shall I proceed? (yes/no)
U: yes
B: Done — your electricity bill payment of 320 EGP (ref 778899) is ready.
→ emits BillPaymentRequest (see §5)
```

### 3.2 English — multi-turn slot filling
```
U: I want to pay my internet bill
B: What's the bill/reference number?
U: 4455123
B: How much would you like to pay?
B: (currency defaults to your usual EGP)
U: 250
B: Please confirm: pay 250 EGP to internet (ref 4455123). Shall I proceed? (yes/no)
U: yes
B: Done — your internet bill payment of 250 EGP (ref 4455123) is ready.
```

### 3.3 Arabic — all slots in one turn
```
U: ادفع فاتورة الكهرباء 320 جنيه رقم 778899
B: تأكيد: دفع 320 جنيه لفاتورة الكهرباء (مرجع 778899). هل أتابع؟ (نعم/لا)
U: نعم
B: تم — تجهيز دفع فاتورة الكهرباء بمبلغ 320 جنيه (مرجع 778899).
```

### 3.4 Arabic — multi-turn
```
U: عايز ادفع فاتورة النت
B: ما هو رقم الفاتورة/المرجع؟
U: 4455123
B: كم المبلغ الذي تريد دفعه؟
U: 250
B: تأكيد: دفع 250 جنيه لفاتورة النت (مرجع 4455123). هل أتابع؟ (نعم/لا)
U: نعم
B: تم — تجهيز دفع فاتورة النت بمبلغ 250 جنيه (مرجع 4455123).
```

---

## 4. Disambiguation: "pay a person" vs "pay a bill"

"pay" is ambiguous. Examples already in the index as **transfers**:
`pay 75 pounds to the landlord`, `ادفع 200 جنيه لصاحب البيت`. Rules (in order):

1. A **biller keyword** (electricity/water/…/كهرباء…) **or** the word
   `bill/فاتورة` **or** a detected **reference number** → `PAY_BILL`.
2. A **person/recipient** after `to/إلى/لـ` with no biller signal → `TRANSFER_MONEY`.
3. Otherwise fall back to the FAISS semantic classifier, then the LLM.

Worked examples:
| Utterance | Intent | Why |
|---|---|---|
| `pay 75 to the landlord` | TRANSFER_MONEY | person, no biller/bill/ref |
| `pay my electricity bill 778899` | PAY_BILL | biller + "bill" + ref |
| `pay 200 water ref 5512` | PAY_BILL | biller + ref |
| `ادفع لصاحب البيت 500` | TRANSFER_MONEY | person, no biller |
| `ادفع فاتورة المياه ٣٠٠` | PAY_BILL | "فاتورة" + biller |

---

## 5. Structured output (what the hub receives)

New schema `BillPaymentRequest`, validated exactly like `TransferRequest`
(amount > 0, known currency, required fields present):

```json
{
  "intent": "pay_bill",
  "biller": "electricity",
  "biller_category": "electricity",
  "reference_number": "778899",
  "amount": "320",
  "currency": "EGP",
  "note": null
}
```
Transfer payload is unchanged:
```json
{ "intent": "transfer_money", "amount": "500", "currency": "USD",
  "recipient": "Ahmed Nassar", "source_account": null, "note": null }
```
> Field names are a placeholder — I'll mirror your hub's exact contract
> (`biller_id` vs `biller`, `bill_number` vs `reference_number`, etc.).

---

## 6. FAISS training examples to add (EN + AR)

```python
# English
("pay my electricity bill", Intent.PAY_BILL),
("pay 320 EGP electricity bill 778899", Intent.PAY_BILL),
("I want to pay the water bill", Intent.PAY_BILL),
("settle my internet bill reference 4455123", Intent.PAY_BILL),
("pay the gas bill number 99100", Intent.PAY_BILL),
("pay my mobile bill for Vodafone", Intent.PAY_BILL),
("can you pay my phone bill", Intent.PAY_BILL),
("pay utility bill 5512 amount 150", Intent.PAY_BILL),
# Arabic
("ادفع فاتورة الكهرباء", Intent.PAY_BILL),
("عايز ادفع فاتورة النت", Intent.PAY_BILL),
("سدد فاتورة المياه رقم ٤٤٥٥", Intent.PAY_BILL),
("ادفع فاتورة الغاز ٣٢٠ جنيه", Intent.PAY_BILL),
("اريد دفع فاتورة الموبايل فودافون", Intent.PAY_BILL),
("ادفع فاتورة الكهرباء 778899 بمبلغ 320", Intent.PAY_BILL),
```
(Index auto-rebuilds via the Active Learning nightly hot-swap, or immediately on
boot since examples are static.)

---

## 7. Entity extraction (biller + reference number)

- `extract_biller(text, lang)` — scan for a category keyword; else take the noun
  after `bill/فاتورة` (e.g. "City Power Co").
- `extract_reference_number(text)` — a digit run (optionally after
  `ref/reference/number/رقم/مرجع`), kept as a **string** (leading zeros matter),
  and only when the intent is `PAY_BILL` so it isn't confused with `amount`.
  - Guard: when both an amount and a reference exist (`pay 320 ... 778899`), the
    token tied to a currency / "amount" cue is the amount; the one after
    `ref/number/رقم` (or the remaining long digit run) is the reference.

---

## 8. Bilingual templates to add

```
ask_biller:     EN "Which bill would you like to pay?"      AR "أي فاتورة تريد دفعها؟"
ask_reference:  EN "What's the bill/reference number?"      AR "ما هو رقم الفاتورة/المرجع؟"
bill_confirm:   EN "Please confirm: pay {amt} {ccy} to {biller} (ref {ref}). Shall I proceed? (yes/no)"
                AR "تأكيد: دفع {amt} {ccy} لفاتورة {biller} (مرجع {ref}). هل أتابع؟ (نعم/لا)"
bill_completed: EN "Done — your {biller} bill payment of {amt} {ccy} (ref {ref}) is ready."
                AR "تم — تجهيز دفع فاتورة {biller} بمبلغ {amt} {ccy} (مرجع {ref})."
```

---

## 9. Optional — auto-alias / habits for bills (follow-up)

Mirror the transfer rules per biller+reference:
- Save a shortcut `electricity` → `{biller: electricity, reference_number: 778899}`
  after N confirmed payments, so "pay electricity 320" skips asking for the ref.
- Recommended as a **follow-up PR** after the core pay-bill flow lands.

---

## 10. Files to touch

| File | Change |
|---|---|
| `app/schemas.py` | `Intent.PAY_BILL`, `BillEntities`, `BillPaymentRequest` (+ validators) |
| `app/config.py` | `biller_categories`, per-intent required slots |
| `app/nlu/examples.py` | bilingual `pay_bill` training examples |
| `app/nlu/entities.py` | `extract_biller`, `extract_reference_number` |
| `app/nlu/pipeline.py` | `validate_bill_payment`; route extraction by intent |
| `app/conversation/state.py` | `biller` + `reference_number` slots; per-intent required set |
| `app/conversation/engine.py` | intent-aware collect/confirm/complete |
| `app/conversation/templates.py` | bilingual bill prompts/confirm/completed |
| `app/static/*.html` | (optional) example bill buttons in the simulator |
| `tests/` | bill flow + disambiguation tests (EN/AR) |

---

## 11. Open decisions (need your confirmation)

1. **Amount source** — customer always states it (recommended), or leave empty for the hub?
2. **Biller ID** — category list + free-text (recommended), or strict list only?
3. **Required slots** — confirm `biller, reference_number, amount, currency`; is `reference_number` ever optional?
4. **Payload field names** — your hub's exact contract (else I use the names in §5).
5. **Bill auto-alias** — now, or follow-up (recommended)?
