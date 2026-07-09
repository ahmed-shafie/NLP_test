# API Integration — How to Call, What to Send, What You Receive

**Service:** NLU + LLM Middleware (bilingual EN/AR banking assistant)
**Audience:** client / mobile developers calling the API.

---

## 1. How to contact the API
- **Protocol:** HTTP(S) REST, JSON.
- **Method:** `POST`
- **Header:** `Content-Type: application/json`
- **Base URL:** `https://<host>` (endpoints are also mirrored under `/v1`, e.g. `/v1/conversation/text`).
- **Authentication:** handled by your channel/gateway — the service assumes the caller is already
  authenticated. Do **not** put credentials in the message body.
- **Two endpoints:**
  - `POST /conversation/text` — **stateful multi-turn chat** (recommended for the app).
  - `POST /nlu/parse` — **stateless** single-message parse (no conversation memory).

---

## 2. `POST /conversation/text` (recommended)

### 2.1 What you send
| Field | Required | Type | Description |
|---|---|---|---|
| `text` | ✅ | string | The user's message this turn. |
| `session_id` | ⭐ | string | Keep the **same value** for every turn of one conversation; use a new one for a new chat. If omitted, a new session is created and returned. |
| `language` | — | string | `"en"` or `"ar"`. Optional hint; auto-detected if omitted. |
| `user_id` | — | string | The authenticated customer id. |

```json
{
  "text": "حول 500 ريال إلى أحمد",
  "session_id": "sess_abc123",
  "language": "ar",
  "user_id": "CUST_10045"
}
```

### 2.2 What you receive
| Field | Type | Description |
|---|---|---|
| `session_id` | string | Echo of the session; store it and reuse next turn. |
| `reply` | string | The text to show the user. |
| `status` | string | `selecting` · `collecting` · `disambiguating` · `confirming` · `completed` · `cancelled`. |
| `language` | string | Detected/used language. |
| `intent` | string \| null | `transfer_money` · `pay_bill` · `small_talk` · `inappropriate` · `fallback`. |
| `pending_slot` | string \| null | The field still needed (prompt the user for it). |
| `complete` | boolean | `true` only when the action is fully collected **and** confirmed. |
| `slots` | object | Collected fields so far (amount, currency, recipient, biller, reference…). |
| `transfer` | object \| null | Validated transfer object when applicable. |
| `bill` | object \| null | Validated bill object when applicable. |
| `flagged_terms` | string[] | Non-empty when the message was moderated. |
| `block_trace` | array | Per-step trace (debugging; can be ignored by the UI). |

```json
{
  "session_id": "sess_abc123",
  "reply": "تمام، تحويل 500 ريال إلى أحمد. أأكد؟ (نعم/لا)",
  "status": "confirming",
  "language": "ar",
  "intent": "transfer_money",
  "pending_slot": null,
  "complete": false,
  "slots": { "amount": 500, "currency": "SAR", "recipient": "أحمد" },
  "transfer": { "amount": 500, "currency": "SAR", "recipient": "أحمد" },
  "bill": null,
  "flagged_terms": [],
  "block_trace": []
}
```

### 2.3 Example multi-turn exchange
```jsonc
// Turn 1 — you send
{ "text": "I want to transfer money", "session_id": "s1" }
// receive: status=collecting, pending_slot="amount", reply="How much...?"

// Turn 2 — you send
{ "text": "500 to Ahmed", "session_id": "s1" }
// receive: status=confirming, reply="Send 500 SAR to Ahmed? (yes/no)"

// Turn 3 — you send
{ "text": "yes", "session_id": "s1" }
// receive: status=completed, complete=true, transfer={amount:500,currency:"SAR",recipient:"Ahmed"}
```

### 2.4 How to drive the flow
- Reuse the **same `session_id`** each turn; show `reply` every time.
- Keep sending the user's next message while `status` is `collecting` / `disambiguating` /
  `confirming`.
- Stop when `complete: true` (status `completed`) — the action is ready — or `cancelled`.
- If `flagged_terms` is non-empty, it was a moderation redirect; just show `reply`.

---

## 3. `POST /nlu/parse` (stateless, optional)

### 3.1 What you send
| Field | Required | Type | Description |
|---|---|---|---|
| `text` | ✅ | string | The message to parse. |
| `language` | — | string | `"en"` / `"ar"` hint; auto-detected if omitted. |
| `account_number` | — | string | If given, resolves the beneficiary by account lookup. |

```json
{ "text": "pay my STC bill 12345, 200 riyals", "language": "en" }
```

### 3.2 What you receive
| Field | Type | Description |
|---|---|---|
| `text` | string | Echo of the input. |
| `language` | string | `en` / `ar`. |
| `intent` | string | The recognised intent. |
| `confidence` | number | 0..1 confidence for the intent. |
| `intent_source` | string | `semantic` / `keyword`. |
| `entities` | object | Extracted slots (amount, currency, recipient, source_account, note…). |
| `resolved_recipient` | object \| null | Best contact-book match. |
| `resolved_beneficiary` | object \| null | Beneficiary resolved by account lookup. |
| `beneficiary_source` | string \| null | `database` / `llm`. |
| `llm_assisted` | boolean | `true` if the LLM fallback filled/corrected slots. |
| `clarification` | string \| null | Follow-up suggestion when the request is incomplete. |
| `block_trace` | array | Per-step trace (optional to use). |

```json
{
  "text": "pay my STC bill 12345, 200 riyals",
  "language": "en",
  "intent": "pay_bill",
  "confidence": 0.95,
  "intent_source": "semantic",
  "entities": { "amount": 200, "currency": "SAR", "recipient": null, "source_account": null, "note": null },
  "resolved_recipient": null,
  "resolved_beneficiary": null,
  "beneficiary_source": null,
  "llm_assisted": false,
  "clarification": null,
  "block_trace": []
}
```

---

## 4. Errors
| HTTP | Meaning | What to do |
|---|---|---|
| `400` / `422` | Invalid/empty body (e.g. missing `text`) | Fix the payload; show a generic error to the user. |
| `413` | Request too large | Shorten the message. |
| `5xx` | Server/model issue | Retry once; otherwise show a friendly fallback message. |

Errors return a JSON body, e.g.:
```json
{ "detail": "text: field required" }
```
