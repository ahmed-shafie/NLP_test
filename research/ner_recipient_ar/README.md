# PoC: Arabic beneficiary-name detection with Hugging Face Transformers

Question asked: *can a Hugging Face model replace the 20k-row name gazetteer that keeps
matching ordinary speech (`الي` registered as a name), and does it survive real Saudi
utterances?*

Answer: **yes, and the decisive ingredient was not the model — it was training it on real
out-of-domain negatives.** The first version scored 100 % on our own data and invented a
beneficiary on 45 % of real Saudi utterances.

Nothing here is wired into the running app: this directory is offline research, and
`app/` is untouched. `results.txt` is the raw output of the run described below.

## Reproduce

```bash
# 1. MASSIVE ar-SA (CC BY 4.0, Amazon Science) -> research/ner_recipient_ar/data/
curl -sSL https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz \
  | tar xz -C /tmp 1.0/data/ar-SA.jsonl 1.0/LICENSE
mkdir -p research/ner_recipient_ar/data
cp /tmp/1.0/data/ar-SA.jsonl research/ner_recipient_ar/data/
cp /tmp/1.0/LICENSE research/ner_recipient_ar/data/MASSIVE-LICENSE

# 2. generated banking utterances + real MASSIVE negatives (train partition only)
python research/ner_recipient_ar/gen_data.py

# 3. fine-tune on CPU (~8 min on 8 cores)
python research/ner_recipient_ar/train_ner.py

# 4. compare every system on every slice
python research/ner_recipient_ar/run_compare.py
```

Data, generated files and the 415 MB checkpoint are git-ignored.

---

## 1. Systems compared

| # | System | What it is |
|---|---|---|
| S1 | **current: regex cue + gazetteer** | `entities.extract_recipient` + `canonicalize_recipient` — `إلى/الى/الي/ل` cue patterns, then spelling correction against `app/data/names.csv` (20 087 rows) |
| S2 | `CAMeL-Lab/bert-base-arabic-camelbert-mix-ner` | off-the-shelf, Apache-2.0, no training |
| S3 | `CAMeL-Lab/bert-base-arabic-camelbert-msa-ner` | off-the-shelf, Apache-2.0, no training |
| S4 | **fine-tuned** `camelbert-mix` + PER head | trained here, 8.1 min on 8 CPU cores |
| S5 | **S4 + beneficiary-directory tie-break** | S4, then the `لـ` reading is disambiguated against the customer's own beneficiary list |

Licences verified from the source, not from memory: CAMeLBERT checkpoints **Apache-2.0**,
MASSIVE **CC BY 4.0** (Amazon Science), `transformers` **Apache-2.0**. No gated model, no
non-commercial clause.

## 2. Evaluation slices

| Slice | Rows | Who wrote it | What it measures |
|---|---|---|---|
| `gold(37)` | 37 | us, before this PoC (`app/eval/nlu_gold.jsonl`) | the Arabic transfer phrasings we already claim to support |
| `hard(20)` | 20 | me, for this PoC (`hard_ar.jsonl`) | name-first / no-cue / dialect / own-account, **8 of them negatives** |
| `massive-neg(400)` | 400 | **nobody here** — MASSIVE `ar-SA`, *test* partition, no `person` annotation | how often a system **invents** a beneficiary on real Saudi speech |
| `massive-per(200)` | 200 | **nobody here** — MASSIVE `ar-SA`, *test* partition, gold `person` slot | can it find a person name in out-of-domain phrasing |

Metric: exact match of the recipient string after `app.nlu.normalize.normalize` (so
`أحمد == احمد`, the same folding the beneficiary lookup uses). Negatives must yield `None`.

MASSIVE is split by **its own partition**: only `train` rows were used for fine-tuning,
only `test` rows for evaluation — no leakage.

## 3. Results

| System | gold(37) | hard(20) | massive-neg(400) invented | massive-per(200) | latency |
|---|---|---|---|---|---|
| S1 current regex + gazetteer | **1.000** | 0.500 | **71 / 400 = 17.8 %** | 0.010 | 1.2 ms |
| S2 camelbert-mix-ner (off the shelf) | 0.432 | 0.700 | 8 / 400 = 2.0 % | 0.605 | 7.9 ms |
| S3 camelbert-msa-ner (off the shelf) | 0.649 | 0.850 | 15 / 400 = 3.8 % | 0.650 | 8.1 ms |
| S4 fine-tuned | 0.946 | 1.000 | 8 / 400 = 2.0 % | 0.750 | 8.5 ms |
| **S5 fine-tuned + directory tie-break** | **1.000** | **1.000** | **8 / 400 = 2.0 %** | **0.760** | 8.5 ms |

`massive-per` understates every system: MASSIVE annotates the attached preposition as part of
the name (gold `لمريم`), which is exactly what we strip. Scoring `مريم` against gold `لمريم` as
correct, S5 reaches **176 / 200 = 0.880**.

`gold(37) = 1.000` for the current system is **not** a compliment: those 37 rows were written
against these regexes. The columns that carry information are `hard` and the two MASSIVE ones.

## 4. The finding that matters

The first fine-tune used only our generated banking sentences plus 30 hand-written negatives:

| version | gold | hard | invented on massive-neg |
|---|---|---|---|
| trained on our data only | 0.919 | 0.950 | **181 / 400 = 45.3 %** |
| + 3 700 real MASSIVE `train` rows as negatives | 0.946 | 1.000 | **8 / 400 = 2.0 %** |

(the accuracy of the first row was measured before the `لل` decoding fix, which is worth a
couple of points there and nothing at all on the invention rate)

Same architecture, same banking accuracy, **22× fewer invented beneficiaries.** A model
trained only on sentences we imagined learns "any unfamiliar token after a preposition is a
name" — literally the gazetteer's bug, relearned. The fix was data, not modelling.

## 5. Error analysis

**The 8 remaining "false positives" are mostly real names that MASSIVE simply did not tag as
`person`** (singers are tagged `artist_name`):

```
خل ارتاح لمحمد عبده جاهزة عشان تشغلها   → pred 'محمد عبده'  (a person; tagged artist_name)
ذكرني بموعدي الثلاثاء مع منى            → pred 'منى'        (a person; gold has no slot)
ملحم بركات                              → pred 'م بركات'    (a person, span cut short)
أرسل email لجدتي يقول بنزورك يوم السبت   → pred 'دتي'        ← a genuine error
```
So the true invention rate is **below 2 %**. Two of the eight are real mistakes, and both are a
span that starts mid-word — which the beneficiary lookup would reject anyway.

The current system's failures on the same slice are not names at all — they are the regex
swallowing whole clauses:

```
كم نحتاج لحين نوصل المطار       → recipient 'حين نوصل المطار'
هل لدي أي طلبات صداقة جديدة     → recipient 'دى أي طلبات صداقة جديدة'
```
and on our own hard slice it produced `recipient='حساب'` for **`حول 500 إلى حسابي`** — an
own-account transfer read as a transfer to a person called "حساب".

## 6. Cost

* model on disk **415 MB** (fp32); ~450 MB RSS when loaded
* latency on 8 CPU cores, short utterances: **mean 7.9 ms, p50 7.8, p95 8.9, max 13.9 ms**
  → inside the 800 ms budget with three orders of magnitude to spare, unlike an LLM
* training: **8.1 min** on CPU, no GPU, 9 700 rows, 3 epochs
* dynamic int8 quantisation would roughly quarter the size; not needed yet

## 7. Honest caveats

1. **`hard(20)` is mine.** I wrote it *and* then tuned the `لـ` decoding while looking at it,
   so 20/20 is optimistic. The MASSIVE columns are the trustworthy ones.
2. **Still no customer data.** MASSIVE is voice-assistant phrasing, not banking. It proves the
   model does not hallucinate on real Saudi speech; it cannot prove the banking error rate.
3. **Confidence interval**: 2.0 % on 400 rows is ±1.4 % at 95 %. Don't quote "2 %" as exact.
4. **Arabic only.** The English path (`_EN_RECIPIENT_RE`) still needs a capital letter, so
   `send 500 to ahmed hassan` fails — untouched by this PoC.
5. The generated grammar is my imagination too. Its role is to teach the *shape* of banking
   requests; the negatives are what keep it honest.

## 8. Recommendation

Integrate as the **primary** Arabic recipient extractor, keeping everything downstream intact:

```
NER (PER span)  →  proclitic readings ("لمحمد" → محمد | لمحمد)
                →  beneficiary directory match (existing resolve_contact)
                →  disambiguation prompt when >1 match
                →  deterministic confirmation ("(yes/no)")   ← unchanged, Tier A
```
* the gazetteer stops being an *extractor* and remains only a spelling helper — the `الي`
  class of bugs disappears by construction;
* keep the regex as the fallback when the model file is absent, exactly like Stanza today;
* no LLM anywhere on this path, no change to money-critical replies;
* ship the model file as an artefact (or an internal HF repo) — 415 MB does not belong in git;
* add `massive-neg` as a permanent eval slice with a hard gate: **invented recipients ≤ 3 %**.

Not recommended: adopting Rasa. It replaces the dialogue framework we already have and would
not have changed a single number in this table.
