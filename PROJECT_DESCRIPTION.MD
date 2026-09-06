# Project Description

**What you're evaluating:** a LangGraph agent that classifies e-commerce customer
support tickets into 5 categories — `order_status`, `refund_request`,
`product_issue`, `account_help`, `other`.

**Full worksheet:** https://docs.google.com/spreadsheets/d/1MVIxuy2C6FNDcDVEgmZjV_LqX9Fd4rCRbpv2qRabeSY/edit?gid=1190010252#gid=1190010252

---

## Known issue: dataset mismatch and an unexplained accuracy gap

Two things came up while working through this exercise that materially affect how
to read the spreadsheet's failure analysis against the notebook's own runs.

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
"keyword anchoring" cluster (`t012`–`t017`, `t024`–`t025`) that Step 4's failure
category is built on. That cluster is genuinely documented in the worksheet's
static data, but a live rerun of the notebook today does not reproduce those
specific failures. The cause isn't confirmed — it could be model drift (the
`gpt-4o-mini` alias isn't perfectly reproducible across checkpoints even at
temperature 0), different sampling settings used for the original run, or the
worksheet's `Predicted Intent` column having been hand-authored to illustrate the
failure taxonomy rather than captured from a live, reproducible call.

**Consequence:** because of this, the failure-category work done in the
spreadsheet (Steps 1–4, based on this worksheet's static `Predicted Intent`
column) does not actually match what a fresh `python -m jupyter nbconvert
--execute` run of the notebook produces today — the live classifier no longer
fails on the same rows the worksheet documents. The prompt fix in Step 4 was
validated directly against the worksheet's documented failure rows (all 8 now
classify correctly), not against a live baseline rerun, since the live baseline
no longer exhibits the bug.

---

## Part 1: Inspect and annotate classifier failures

- Open **Step 1 · Tag failures**. Filter `Pass / Fail` to `Fail`.
- Don't edit ticket text, true category, predicted category, reasoning, or `Pass / Fail`.
- In `Annotation`, explain *why* the model got each row wrong.
- A strong annotation names the competing cue, the wrong route taken, and the route that should have won — not just "wrong" or "confusing."

## Part 2: Create failure categories

- Open **Step 2 · Group + name**. Re-read your Step 1 annotations and cluster failures that fail the same way. Group first, name second.
- For each cluster, record: a short title (e.g. `[Signal]_Missed`), a one-line definition of the exact failure pattern, a count of affected rows, and one sentence on why it matters.
- Open **Step 3 · Label failures** and assign every failed row to one of the Step 2 categories, using identical spelling so filters/summaries work.

## Part 3: Tweak the classifier prompt

- Open **Step 4 · Tweak the prompt**. Pick the failure category that matters most for the business and make **one** focused prompt change — not a full rewrite.
- Options: a disambiguation rule separating the failure pattern from what it's confused with, a few-shot example of the corrected route, or a regression check re-running previously-passing tickets to confirm no overcorrection.
- Use notebook Sections 9–10 to rerun the classifier and compare v1 vs v2, looking for both wins and regressions.

## Part 4: Run LLM-as-a-judge (Optional)

- Run from notebook Section 11 after Step 3 labels are complete.
- The judge doesn't discover the failure category — you pick one target category from Step 2/3 first, then the judge answers one binary question per failed row: does this row belong to the target category?
- Key fields: **Rows judged** (all 29 classifier-failure rows), **Target category** (your pick), **Ground Truth** (TRUE if the row's Step 3 label matches the target category), **Judge Prediction** (TRUE/FALSE + reason), **Agreement** (% where Judge Prediction matches Ground Truth).
- After running, **Step 5 · LLM-J** should hold the failure category, ground truth, judge prediction, judge reasoning, and match indicator for every row.

## Part 5: Align the LLM judge (Optional)

- Open **Step 6 · Tweak the LLM judge** after Step 5 is populated. Inspect rows where `Match?` is false.
- First move: compare at least two candidate judge models on the same binary judge task, and pick whichever's mistakes are easiest to defend or fix.
- If a different/stronger model fixes most misses, prefer swapping the model over patching the prompt. If multiple models share the same mistake, calibrate the prompt instead (a contrast rule or a representative few-shot example).
- Rerun notebook Section 11 after any model or prompt change and compare agreement across runs.

## Part 6: Inspect evaluation traces with LangSmith

- Run notebook Section 2 before the classifier so LangSmith tracing is enabled and the project is configured.
- In Sections 5a–6, confirm each ticket classification appears as an individual trace in the LangSmith project.
- Use the ticket/example ID in the run metadata to connect a trace to its spreadsheet/CSV row, and the run name + prompt version metadata to separate baseline vs improved runs.
- When reviewing failures or regressions, inspect the true category, predicted category, correctness flag, and model reasoning alongside the trace.

---

## Evaluation criteria

| Step | Requirement |
|---|---|
| Step 1 | Filter `Pass / Fail` to `Fail`. Add a specific annotation per failed row — Step 2 clustering depends on these. |
| Step 2 | Cluster Step 1 annotations into 2–4 named failure categories, each with a title, one-line definition, and one-line "why it matters." |
| Step 3 | Assign every failed row to a Step 2 category, turning the clusters into a labeled failure dataset. |
| Step 4 | Pick the highest-impact failure category and make ONE focused prompt change. Re-run the notebook. Record expected impact and regression risk. |
| Optional LLM-J | Notebook Section 11 runs three judge prompts and writes agreement-with-humans scores into `Step 4 · LLM-J`. `Step 4 · Judge disagreements` shows where you and the winning judge disagree — that's where alignment work happens. |
