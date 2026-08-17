---
name: testing-slots-and-topic-answers
description: Runtime-test slot extraction (recipient/amount/reference/biller/account words), Arabic proclitic/article handling, memory shortcut matching, the topic-answer reply family, cancel vocabulary and the insufficient-funds/offered-balance flow in the /assistant UI, including a "before" build for A/B. Use when verifying changes to app/nlu/entities.py, app/nlu/normalize.py, app/data_loader.py, app/memory/service.py, app/orchestration.py slot grounding, app/conversation/engine.py, topic_replies.py, templates.py, or banking-core service.py.
---

# Testing slot extraction and topic answers

Two claims usually under test together:

1. **Slots**: a payee named with no `to` ("pay Mona 40 riyals") resolves to a real beneficiary, while a
   biller or non-person ("pay mobily 100", "send someone 50", "transfer money 300") never becomes a payee;
   a cue-less number is an amount only where a reference number cannot be.
2. **Topic answers**: the retrieved topic family is corrected from the question's own words (a card that
   "doesn't work" is answered as blocked-card, not as a theft), **without** loosening the confidence gate.

## Setup

- Postgres (`docker compose up -d db banking-core`; host port may be remapped — check `docker compose ps`),
  Banking Core `:8100`, app `uvicorn app.main:app --port 8001`.
- After a machine restart the compose stack usually comes back on its own (postgres `:55432`, redis
  `:56379`, banking-core `:8100`, containerised app `:8000`) — run `docker compose ps` before starting
  anything. Only the branch-under-test app has to be started from source; a known-good command is:

  ```bash
  cd /home/ubuntu/repos/NLP-test && nohup env \
    NLU_LLM_ENABLED=false NLU_REPLY_VARIATION_ENABLED=false \
    NLU_BANKING_CORE_ENABLED=true NLU_BANKING_CORE_URL=http://localhost:8100 \
    NLU_BENEFICIARY_LOOKUP_ENABLED=true \
    NLU_BENEFICIARY_DB_URL='postgresql+psycopg://banking:banking@localhost:55432/banking_core' \
    NLU_SESSION_BACKEND=redis NLU_REDIS_URL=redis://localhost:56379/0 NLU_MEMORY_CACHE_BACKEND=redis \
    .venv/bin/uvicorn app.main:app --port 8001 > /tmp/app8001.log 2>&1 &
  ```

  Startup takes ~20-30 s (spaCy/stanza/FAISS load) before `/health` answers.
- UI: `http://localhost:8001/assistant?dev=1` — `?dev=1` auto-enables Developer mode, so the
  "🔍 Inspect & Debug" panel with the slot table + BLOCK_TRACE is visible without clicking anything.
- Always run with `NLU_LLM_ENABLED=false NLU_REPLY_VARIATION_ENABLED=false`, so any wording difference is
  a real behaviour change and not phrase rotation.
- The env vars that matter are all `NLU_`-prefixed and must be set on the uvicorn process; if an app
  instance is already running, read what it was started with via
  `tr '\0' '\n' < /proc/<pid>/environ | grep NLU_` rather than assuming.
  Beneficiary lookup needs `NLU_BENEFICIARY_LOOKUP_ENABLED=true` +
  `NLU_BENEFICIARY_DB_URL=postgresql+psycopg://banking:banking@localhost:<hostport>/banking_core`, and the
  Banking Core path needs `NLU_BANKING_CORE_ENABLED=true` (default is **false** — without it no preflight,
  so no funds check at all) plus `NLU_BANKING_CORE_URL`.
- Postgres creds default to `banking:banking`, database `banking_core` (see `docker-compose.yml`).
- Health endpoint is `/health` (**not** `/healthz`).
- Seeded owner is `demo`; beneficiary IBAN column in Postgres is `account` (not `account_number`).

## Testing a banking-core change: the docker image goes stale

`docker compose up -d banking-core` keeps serving the **image that was built earlier**, so a change under
`banking-core/` will not appear on `:8100` even though the container is healthy. Symptom: the API still
returns the old payload (e.g. `warnings:["low_funds: …"], blocking:[]` instead of
`blocking:["insufficient_funds: …"]`). Check before trusting it:

```bash
curl -s -X POST localhost:8100/preflight/transfer -H 'content-type: application/json' \
  -d '{"owner_user":"demo","amount":"66635","currency":"SAR"}'
```

(note the field is `owner_user`, not `user_id`). Rather than rebuilding, run the branch's core from source
on a spare port and point the app at it — this also hands you a free A/B pair, since the stale container is
exactly the "before" core:

```bash
cd banking-core && ../.venv/bin/uvicorn banking_core.main:app --port 8101   # after
# app AFTER  -> NLU_BANKING_CORE_URL=http://localhost:8101
# app BEFORE -> NLU_BANKING_CORE_URL=http://localhost:8100   (old container)
```

## Getting a real "before" build

If the branch is based directly on `main`, `git diff main...HEAD` can look like the whole repo when local
`main` is a bootstrap commit. Use a detached worktree instead:

```bash
git worktree add --detach /tmp/before origin/main
cd /tmp/before && /home/ubuntu/repos/NLP-test/.venv/bin/uvicorn app.main:app --port 8002
```

Both builds can share the same Postgres and Banking Core, so the only difference between the two browser
tabs is the port — strong A/B evidence. Note the before build starts with empty memory ("Transfers
learned 0"), which is expected and not a bug.

Pick the baseline deliberately and **say which one you used**: `origin/main` may lag by an already-merged
PR, in which case `HEAD~1` is the only ref that isolates the PR under test. `git log --oneline origin/main
-1` vs `git log --oneline HEAD~1 -1` before deciding; re-point an existing worktree with
`cd /tmp/before && git checkout --detach <sha>`.

## Testing the LLM grounding guard (LLM ENABLED)

Some fixes are specifically about the local model authoring financial slots, and are invisible with the LLM
off. Run a **second pair** of instances with `NLU_LLM_ENABLED=true` (keep
`NLU_REPLY_VARIATION_ENABLED=false`), e.g. `:8003` = after, `:8004` = before, and drive them in the UI —
the Inspect panel header shows `llm: yes`, and `✓ llm_fallback` appears in BLOCK_TRACE, which is how you
prove the model actually ran and was still refused. Budget ~10-40 s per turn on Ollama `qwen2.5:3b`, so
allow long waits before screenshotting.

The model is **nondeterministic**: the invented figure differs run to run (an account `٠١١١٥٥٥٢٤٢` came
back as `115524` in the user's session and `1155242` here). Assert "no amount at all", never a specific
wrong number.

## Testing the insufficient-funds / offered-balance flow

Seeded `demo` balances live in Postgres `accounts` (current `ACC-001` = 12,300.00 SAR — check with
`PGPASSWORD=banking psql -h localhost -p <hostport> -U banking -d banking_core -c 'select * from accounts;'`),
so pick an amount above the current balance to trigger the refusal.

What to assert in the UI: the refusal turn must show **no "Transfer · review" card at all** (the card is the
only on-screen proof there was no confirmation), status `collecting`, pending slot `amount`. Then
`yes`/`نعم` shows a review card at the *offered* figure and BLOCK_TRACE reads
`✓ orchestrator (accept_offered_amount)` — that annotation is the cheapest proof the offer path ran rather
than the original amount. Paths worth covering because they are all distinct code: accept, decline
(→ `cancelled`, side panel emptied), offer expiry (an unrelated turn such as `كم رصيدي` shows
`orchestrator (balance_aside)` and consumes the offer, so a later `yes` re-blocks), and re-typing the
unaffordable figure (must re-block).

## Testing cancel vocabulary and language stickiness

Each cancel word needs its own fresh confirmation screen (**New chat** → transfer → word), since the first
one cancels the flow. Pass = status `cancelled` **and** the "In this transaction" side panel returns to
"No active transfer or bill yet". Then check a post-cancel `yes` does not resurrect anything (expect the
generic menu, status `selecting`).

For language stickiness, send a Latin-script cancel word (`cansel`) inside an Arabic conversation: the
Inspect header must still read `lang: ar` and the reply must be the Arabic cancellation. Also probe a real
Arabic non-cancel word (e.g. `بزاف`) — it must leave amount/recipient untouched and re-ask, not become a
payee.

## Testing memory shortcuts and bilingual labels

The `/assistant` right-hand panel has full shortcut CRUD ("Quick actions" list with a 🗑 per row, and an
"Add / update shortcut" row of name/amount/currency/recipient inputs + **Save shortcut**), so the whole
shortcut test can be driven in the UI — no API calls needed. The panel re-flows after **New chat** (the
"In this transaction" card collapses), so re-screenshot before clicking the inputs; the recipient input
sits at the right edge and may be partly clipped at 1024px wide but is still clickable.

Behaviour worth asserting explicitly:

- **A shortcut outranks the biller gazetteer.** With a shortcut named `rent` saved, `ادفع الايجار` becomes
  a *transfer* to that shortcut's recipient; with no such shortcut it is the `إيجار` bill (SADAD 153).
  Both are correct — decide which one you are testing and delete/add the shortcut accordingly, otherwise
  a leftover shortcut silently masks the biller test (and vice-versa).
- Cross-language labels only apply to a small hand-written table (`_LABEL_GROUPS` in
  `app/memory/service.py`). Test one pair in each direction (`rent` ↔ `ادفع الايجار`,
  `راتب` ↔ `pay the salary`) plus two negatives: a *different* label (`ادفع الراتب` must not fire `rent`)
  and a *person* label (`mona` must not fire on `ادفع الايجار`).
- Clean up the shortcuts you add (`DELETE /memory/<user>/shortcuts/<name>`, URL-encode Arabic names) so the
  next run starts from the seeded four (`ahmed`, `ahmed-500sar`, `محمد`, `محمد-250sar`).

## Testing the IBAN typo / override flow (add-beneficiary)

Entry is `Add a new beneficiary` / `أريد إضافة مستفيد جديد` → name → account, so the whole flow is typed
into the composer. Useful constants (`tests/test_add_beneficiary.py`): **GOOD**
`SA0380000000608010167519`, **SWAP** `SA0380000000608010167591` (checksum fails, unique adjacent
transposition → "characters 23 and 24 look swapped"), **AMBIG** `SA0380000000608010167516` (checksum fails,
4 candidate repairs → "one character is wrong, though I can't tell which").

What to assert: a failed checksum offers `I'm sure` / `no` and never proposes a corrected IBAN; an aside
(`what is my balance`) answers *and re-emits* the same warning in one turn; only an explicit insist (or a
bare `yes`) reaches a confirmation, and that confirmation carries `⚠️ This IBAN failed its checksum; I'll
use it exactly as you typed it.`; a length/shape failure (`SA1122330000007777`, `1234`, `abcd`) must stay a
hard refusal even when followed by `I'm sure` (the insist then reads as "not an account"); retyping a valid
IBAN clears the override (confirmation with **no** ⚠️ note).

Gotchas:

- The write is guarded by a **unique account** constraint in Banking Core, so re-running a completed
  override with the same IBAN returns `I couldn't add "X": A beneficiary with that account already exists.`
  Since `cc5331c` that refusal reports status **`failed`** (a terminal status alongside
  `completed`/`cancelled`); a chip reading `completed` on such a reply is a bug. Check
  `banking-core/banking_core.db` first and pick a fresh name+IBAN pair if you want to prove a real write.
- Manufacture a **fresh** checksum-invalid IBAN instead of reusing the shared constants (they get written
  by earlier runs, including by API probes): take the GOOD IBAN, swap two adjacent characters, keep the
  ones whose `app.nlu.accounts.analyze_iban_typo(...)` returns a `swapped=(i, i+1)` hint (so the reply
  names a position) and that are absent from the sqlite table. A fresh **valid** IBAN for the
  "successful add still says completed" case can be brute-forced with the mod-97 check.
- Arabic wording of the same flow: warning
  `الآيبان ما ضبط في التحقق الحسابي — يبدو أن الخانتين N وN+1 متبادلتان … أو اكتب "أنا متأكد" … أو "لا" للإلغاء.`,
  override token `أنا متأكد`, confirmation note `⚠️ الآيبان ما نجح في التحقق الحسابي وهنستخدمه زي ما كتبته.`,
  success `تمّ ✅ — أضفت … لقائمة المستفيدين.`
- Probing the flow through `/conversation/text` **writes to the database** just like the UI does — probe
  with throwaway names/IBANs, or the account you meant to use in the recorded run will be duplicate-blocked.
- Ground truth for "stored byte-identically" is the sqlite row:
  `select name, account from beneficiaries where name='…'` — assert `len(account) == 24` and equality with
  what was typed. The audit line is in the app log:
  `saved with an IBAN the customer confirmed against a failed mod-97 check`.
- These replies are in `phrasing.CRITICAL_REPLIES`, so they are never LLM-rewritten; wording differences are
  real behaviour changes.

## Reading the evidence

- BLOCK_TRACE names the topic decision explicitly: `✓ topic_answer (card_blocked @ 0.9031)` vs
  `– topic_answer (no confident topic @ 0.9213)`. That line is the cheapest proof of which family
  answered and whether the gate refused.
- Slot table distinguishes `amount` from `reference` — that is the assertion for reference-vs-amount tests.
- Priced vs unpriced bill numbers (`_priced_runs` / `_MONEY_MARKED_RE` in `app/nlu/entities.py`): a digit
  run with a currency word/symbol against it is the **amount** (`pay my mobily bill 100 sar`,
  `ادفع فاتورة موبايلي ١٠٠ ريال` → Amount 100.00 SAR, Reference —, next prompt asks for the bill number),
  while a bare number after a bill cue stays the **reference** by design (`pay my mobily bill 100`,
  `pay my gas bill 5566` → asks the amount). Do not grade the unpriced case as a bug. The discriminator
  worth recording is `pay stc bill 4455 210 sar` → Reference 4455 **and** Amount 210.00 SAR together.
- A topic answer must leave `amount`/`recipient`/`biller` empty and status `selecting`.

## The gate refusal trap

A decisive cue in the question does **not** entitle it to an answer: retrieval must still be confident and
agreeing. Short phrasings like `my card is not working` / `بطاقتي ما تشتغل` fall to
`– topic_answer (no confident topic)` and return the generic "(1) send money or (2) pay a bill" menu, while
longer ones (`i can't use my card anymore`, `أظن إني ماني قادر استخدم بطاقتي`) do get answered. Report a
refusal as "gate refused, by design" rather than a cue failure — but say clearly that the exact requested
phrasing was not answered, since a user asking for that phrasing usually expects an answer.

Since PR #37 the gate is **per language**: English uses `topic_reply_top_k_en=7` /
`topic_reply_unanimous_threshold_en=0.78` (cross-lingual retrieval scores ~0.1 lower), Arabic keeps
0.74/0.94. So the same score can be accepted in English and refused in Arabic — read the score *and* the
`lang:` chip together. Post-#37, `my card is not working` is answered (`✓ topic_answer (البطاقة لا تعمل @
0.9213)`) while `freeze my card`, `my card is blocked and i cannot use it` and
`what is the exchange rate for usd to sar` are still refused (`no confident topic @ 0.8422`) — expected, not
a bug. Always pair a gate-widening test with executable negatives (`send 100 sar to ahmed`,
`pay my mobily bill 100`, `what is my balance`, `transfer 50 to mona`): a wider gate must never answer an
executable request as a topic.

## Arabic input

Typing Arabic with the computer tool drops characters. Put the string on the clipboard and paste:

```bash
printf 'بطاقتي اتسرقت' | DISPLAY=:0 xclip -selection clipboard
```
then click the composer and press `ctrl+v`, `Return`.

## Money determinism check

Complete one transfer (`pay Mona 40 riyals` → `yes`) and compare the review card with the completion
reply and the side panel: amount, currency, beneficiary name and full IBAN must be identical, with Latin
digits. Then confirm Banking Core actually saw it: `docker compose logs banking-core | grep preflight`
should show `POST /preflight/transfer ... 200`. If the core is being run from source instead (see above),
grep its uvicorn log: `grep POST /tmp/core_after.log`. Any change to those fields blocks the PR.

## Quick probe script before touching the browser

The UI is the deliverable, but `POST /conversation/text` with `{user_id, session_id, text}` returns
`reply`, `status` and `slots`, so a short Python/curl loop can confirm every case is reachable and has a
distinct before/after signature *before* recording. This turns the browser pass into confirmation rather
than exploration, and keeps the recording short. Use the same `session_id` across turns to continue a
conversation.

## Devin Secrets Needed

None — everything runs locally with the local model disabled.
