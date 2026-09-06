# Project Description

**What was evaluated:** a LangGraph agent that classifies e-commerce customer
support tickets into 5 categories — `order_status`, `refund_request`,
`product_issue`, `account_help`, `other`.

**Full worksheet:** https://docs.google.com/spreadsheets/d/1MVIxuy2C6FNDcDVEgmZjV_LqX9Fd4rCRbpv2qRabeSY/edit?gid=1190010252#gid=1190010252

---

## Part 1: Inspect and annotate classifier failures

**What I did:** Filtered the worksheet's Step 1 tab to `Fail` and annotated each
failed row with the specific cue the model latched onto versus what the customer
actually needed.

**Findings:** Two distinct failure directions kept showing up, both driven by the
same root cause — the classifier routing on incidental nouns in the ticket
("order," "delivered," "package," "address") instead of the customer's actual
requested action:

- **6 of 8 tickets:** an explicit refund ask (e.g. *"I never received my order and
  I want my money back"*) got overridden by delivery/order language and misrouted
  to `order_status`.
- **2 of 8 tickets:** a genuine non-delivery issue got misrouted to `account_help`
  because the ticket happened to mention the word "address" (e.g. *"Carrier
  delivered to the wrong address"*), even though the customer never asked to
  update anything on their account.

## Part 2: Create failure categories

**What I did:** Clustered the Step 1 annotations and picked one category to carry
into Step 4: **"Delivery/order keyword anchoring overrides the customer's actual
stated need (refund_request or order_status)."**

**Findings:** This cluster was the highest-impact one to fix, for two concrete
business reasons — refund-ask tickets misrouted to `order_status` delay the
refund and carry chargeback/dispute risk, and non-delivery tickets misrouted to
`account_help` mean a real missing package sits uninvestigated while the customer
is asked to "confirm their address" instead of getting an actual answer.

## Part 3: Tweak the classifier prompt

**What I did:** Iterated through several versions of the disambiguation rule
before landing on a final prompt. Early attempts either flipped the priority the
wrong way (making refund-ask always win, which regressed the model against a
*different* dataset's labeling convention) or restored the original "non-delivery
wins" rule (which regressed against *this* worksheet's convention once the
underlying dataset was corrected — see the data-mismatch note below). The final
prompt reframes classification around "what action is the customer asking you to
take," with ordered disambiguation rules (explicit refund ask wins over delivery
language; a mentioned address alone doesn't trigger `account_help`) plus three
worked few-shot examples.

**Findings:**
- Validated against the exact 8 documented failure tickets (`t012`–`t017`,
  `t024`–`t025`): **8/8 now classify correctly** — both failure directions from
  Part 1 are fixed.
- Re-running the full 150-ticket golden dataset: baseline `CLASSIFIER_PROMPT`
  scores **96.67%** (145/150); the new prompt scores **96.00%** (144/150) — a
  net *regression* of 1 ticket, caused by two failures unrelated to the targeted
  category: `t006` ("tracking link just spins forever," true `order_status`) and
  `t100` ("2FA codes showing up unrequested," true `account_help`) both got
  misrouted to `other`. The new prompt's "state an explicit action" framing is
  strict enough that these two tickets — which imply a request rather than state
  one outright — no longer clearly match any category. This wasn't in scope to
  fix (the instruction was to fix one category at a time), but it's a real
  trade-off worth knowing about before shipping this prompt as-is.

## Part 4: Run LLM-as-a-judge

**What I did:** Built `llm_as_judge.py`, which reads the worksheet's own
`LLM-J Dataset.csv` (the 29 failed rows) and, per row, runs two independent LLM
calls: a 4-way failure categorizer (`REFUND_MISSED` / `ORDER_STATUS_MISSED` /
`ACCOUNT_HELP_CONFUSION` / `OTHER`) and the worksheet's own `Refund_Missed`
binary judge prompt from Step 6 — shown only the ticket text and predicted
intent, blind to the categorizer's own output.

**Findings:** Across all 29 rows: 12 `REFUND_MISSED`, 9 `ACCOUNT_HELP_CONFUSION`,
6 `OTHER`, 2 `ORDER_STATUS_MISSED`. The independent judge agreed with the
categorizer's implied ground truth on **100%** of rows — every row the
categorizer labeled `REFUND_MISSED` also got a `TRUE` from the standalone judge,
and vice versa. See `LLM-J-gpt-4o.csv` / `LLM-J-gpt-5-mini.csv` for the raw
per-row output.

## Part 5: Align the LLM judge

**What I did:** Ran the identical judge prompt through two candidate models,
`gpt-4o` and `gpt-5-mini`, and compared them row by row (`LLM-J-comparison.csv`).

**Findings:** Both models scored **100% agreement** with ground truth, and
**100% agreement with each other** on every row — no `Match? = false` rows
turned up to align on with this particular 29-row set. That's a real result, not
a null one: it means this specific curated failure set is unambiguous enough that
model choice didn't matter here. Given the tie, `gpt-4o` is the more defensible
default (more established structured-output track record for this kind of
classification judge), but the exercise's usual "which model's mistakes are
easier to fix" tradeoff never came up because neither model made any mistakes on
this set. A noisier or larger judge dataset would be needed to actually surface a
disagreement worth calibrating.

## Part 6: Inspect evaluation traces with LangSmith

**What I did:** Debugged and fixed a LangSmith authentication bug before this
part could work at all — the notebook's credential-loading helper only prompted
for a new API key `if` the environment variable wasn't already set, so a stale or
revoked key left over in a running kernel process silently blocked every trace
upload with `403 Forbidden`, regardless of how many times the key was rotated in
`.env.demo` or `langsmith_key.txt`. Fixed the notebook so it always loads
`LANGSMITH_API_KEY`/`OPENAI_API_KEY` from `.env.demo` when present, overriding
any stale in-process value. Then ran the notebook end-to-end
(`jupyter nbconvert --execute`) to confirm traces actually export.

**Findings:** After the fix, the full notebook run produced zero `403`/export
errors. Each ticket classification appears in the LangSmith project
`customer-support-evals` as its own trace, tagged via OpenTelemetry span
attributes (`eval.run_name`, `eval.prompt_version`, `eval.example_id`,
`eval.true_category`, `eval.predicted_category`, `eval.correct`,
`eval.reasoning`) — so any row in `results_v1.csv`/`results_v2.csv` can be traced
back to the exact model call that produced it, and baseline vs. improved runs are
separable by `otel_run_name`/`otel_prompt_version`.

---

## Known issue: dataset mismatch and an unexplained accuracy gap

Two things came up while working through this exercise that materially affect how
to read the findings above.

**1. The notebook's original dataset didn't match this worksheet.**
`week4_customer_support_evals.ipynb` originally generated its own ticket dataset
from a hand-written `SEED_TICKETS` list (25 seeds, expanded via LLM-generated
paraphrases). That dataset used a *different* ground-truth convention than this
worksheet — e.g. "I never received my order and I want my money back" was labeled
`order_status` in the notebook's seed data, but `refund_request` in this
worksheet's golden dataset. Since the two disagreed on the correct label for the
same ticket, scoring the notebook's classifier against its own seed data was not
measuring the same thing this worksheet's Step 1–4 exercise was built around.
**Fix:** the notebook's `SEED_TICKETS` + paraphrase-generation cell were replaced
with the worksheet's own 150-ticket golden dataset (pulled directly from this
sheet), so `true_category` now matches this worksheet's labels exactly.

**2. The worksheet's reported 80.70% overall accuracy does not reproduce live.**
Re-running the identical `CLASSIFIER_PROMPT` text, against the identical ticket
text, live through `gpt-4o-mini` at `temperature=0` today scores **96.67%**, not
80.70%. Comparing row-by-row, the worksheet's own `Predicted Intent` column agrees
with a fresh live run on only 83.3% of rows — including the exact 8-ticket
"keyword anchoring" cluster (`t012`–`t017`, `t024`–`t025`) that Part 2/3's failure
category is built on. That cluster is genuinely documented in the worksheet's
static data, but a live rerun of the notebook today does not reproduce those
specific failures. The cause isn't confirmed — it could be model drift (the
`gpt-4o-mini` alias isn't perfectly reproducible across checkpoints even at
temperature 0), different sampling settings used for the original run, or the
worksheet's `Predicted Intent` column having been hand-authored to illustrate the
failure taxonomy rather than captured from a live, reproducible call.

**Consequence:** because of this, the failure-category work described in Parts
1–3 (based on this worksheet's static `Predicted Intent` column) doesn't line up
with what a fresh notebook run produces today — the live classifier no longer
fails on the same rows the worksheet documents. That's why Part 3's prompt fix
was validated directly against the worksheet's documented failure rows (8/8 now
correct) rather than against a live baseline rerun, which no longer exhibits the
original bug.
