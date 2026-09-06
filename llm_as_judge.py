"""
LLM-as-a-judge for the customer-support classifier's failed traces.

Reads "LLM-J Dataset.csv" (the 29 failed classifier traces exported from the course
evaluation worksheet: https://docs.google.com/spreadsheets/d/1MVIxuy2C6FNDcDVEgmZjV_LqX9Fd4rCRbpv2qRabeSY).
Each row has: Test ID, User Input, Ground Truth Intent, Predicted Intent.

For each row, two independent LLM calls are made:

1. A failure-category classifier: given the ticket, the ground-truth intent, and the
   classifier's (wrong) predicted intent, bucket the failure into one of
   REFUND_MISSED / ORDER_STATUS_MISSED / ACCOUNT_HELP_CONFUSION / OTHER.
   -> "Failure Category" column.
   "Ground Truth" (bool) is then just Failure Category == "REFUND_MISSED" -- the
   target category this judge is being aligned on.

2. A narrow, single-purpose judge -- shown ONLY the User Input and Predicted Intent
   (not the failure category from step 1) -- that answers one yes/no question:
   "Is this classifier mistake an instance of a missed refund request?"
   -> "Judge Prediction" (bool) + "Judge Reasoning".
   This prompt is the worksheet's own completed Refund_Missed judge prompt
   (Step 6), used verbatim.

"Match?" is True when the two independent calls agree (Ground Truth == Judge
Prediction). Because the two calls are independent, disagreement is expected and is
the whole point of the exercise -- it's what tells you the judge prompt needs
tightening.

Run the whole pipeline once per model (gpt-4o and gpt-5-mini by default), writing
"LLM-J-<model>.csv" for each, then print a side-by-side comparison of the two runs.
"""

import os
import sys
import getpass
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

INPUT_CSV = "LLM-J Dataset.csv"
MODELS_TO_RUN = ["gpt-4o", "gpt-5-mini"]

FAILURE_CATEGORIES = ["REFUND_MISSED", "ORDER_STATUS_MISSED", "ACCOUNT_HELP_CONFUSION", "OTHER"]


# --------------------------------------------------------------------------
# 0. Credentials (same pattern as week4_customer_support_evals.ipynb: prefer
#    .env.demo, but never let a stale process env var block a fresh key)
# --------------------------------------------------------------------------
def _load_dotenv(path: str) -> dict:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    values = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def _set_api_key():
    dotenv = _load_dotenv(".env.demo")
    key = "OPENAI_API_KEY"
    if dotenv.get(key):
        os.environ[key] = dotenv[key]
    elif not os.environ.get(key):
        os.environ[key] = getpass.getpass("OpenAI API key: ")


# --------------------------------------------------------------------------
# 1. Failure-category classifier (multi-class: which kind of misroute is this?)
# --------------------------------------------------------------------------
FAILURE_CATEGORY_PROMPT = ChatPromptTemplate.from_template("""\
You are reviewing a failed prediction from an e-commerce support-ticket classifier.
The classifier predicted the WRONG intent for this ticket. Your job is to bucket
WHY it failed into exactly one of these categories:

- REFUND_MISSED: the customer explicitly asked for a refund, their money back,
  reimbursement, a credit, or a charge reversal, and the classifier's predicted
  intent is NOT refund_request.
- ORDER_STATUS_MISSED: the customer's real issue is about where an order is,
  tracking, delivery ETA, or non-delivery, and the classifier's predicted intent
  is NOT order_status.
- ACCOUNT_HELP_CONFUSION: the failure involves account_help being confused with
  another category -- either the ground truth is account_help and the classifier
  missed it, or the classifier wrongly predicted account_help for a ticket that
  isn't actually about login/password/profile/payment-method-on-file management.
- OTHER: any failure that doesn't clearly fit one of the buckets above.

If more than one bucket could apply, pick the one that matches the Ground Truth
Intent (i.e. what the customer actually needed) rather than the incidental wording
of the ticket.

User Input: {user_input}
Ground Truth Intent: {ground_truth_intent}
Predicted Intent: {predicted_intent}

Return structured output with fields: failure_category, reasoning.
""")


class FailureCategoryResult(BaseModel):
    failure_category: Literal[
        "REFUND_MISSED", "ORDER_STATUS_MISSED", "ACCOUNT_HELP_CONFUSION", "OTHER"
    ]
    reasoning: str = Field(description="One short sentence explaining the bucket choice.")


# --------------------------------------------------------------------------
# 2. Independent binary judge -- the worksheet's own completed Refund_Missed
#    judge prompt (Step 6), used verbatim. Deliberately shown ONLY User Input
#    + Predicted Intent, not the failure category from step 1 -- it must
#    reach its own independent verdict.
# --------------------------------------------------------------------------
JUDGE_PROMPT = ChatPromptTemplate.from_template("""\
You are evaluating an e-commerce support classifier.

Failure category: Refund_Missed
Definition: Refund, credit, charge-reversal, money-back, or post-purchase
price-adjustment intent routed by surrounding order/account/policy context instead
of refund intent.

Given User Input and Predicted Intent, decide whether the classifier mistake is an
instance of this failure category.
Return TRUE only when the user asks for or asks about a post-purchase refund,
money back, credit, price adjustment, charge reversal, refund difference, or
refund eligibility for their order and the Predicted Intent is not refund_request.
Return FALSE for correct predictions, product issues without refund intent,
general policy or pre-purchase questions with no owned order/payment, and all
other failure categories.

User Input: {user_input}
Predicted Intent: {predicted_intent}

Return structured output with fields: label, reasoning.
""")


class JudgeResult(BaseModel):
    label: bool = Field(description="TRUE if this is a REFUND_MISSED instance, else FALSE.")
    reasoning: str = Field(description="One short sentence explaining the verdict.")


# --------------------------------------------------------------------------
# 3. Run the full pipeline once for a given model
# --------------------------------------------------------------------------
def run_judge_for_model(model_name: str, df: pd.DataFrame) -> pd.DataFrame:
    llm = ChatOpenAI(model=model_name, temperature=0)
    category_chain = FAILURE_CATEGORY_PROMPT | llm.with_structured_output(FailureCategoryResult)
    judge_chain = JUDGE_PROMPT | llm.with_structured_output(JudgeResult)

    rows = []
    total = len(df)
    for i, row in df.iterrows():
        print(f"  [{model_name}] {i + 1}/{total}: {row['Test ID']}", file=sys.stderr)

        cat_result = category_chain.invoke({
            "user_input": row["User Input"],
            "ground_truth_intent": row["Ground Truth Intent"],
            "predicted_intent": row["Predicted Intent"],
        })
        ground_truth = cat_result.failure_category == "REFUND_MISSED"

        judge_result = judge_chain.invoke({
            "user_input": row["User Input"],
            "predicted_intent": row["Predicted Intent"],
        })

        rows.append({
            "Test ID": row["Test ID"],
            "User Input": row["User Input"],
            "Ground Truth Intent": row["Ground Truth Intent"],
            "Predicted Intent": row["Predicted Intent"],
            "Failure Category": cat_result.failure_category,
            "Ground Truth": ground_truth,
            "Judge Prediction": judge_result.label,
            "Judge Reasoning": judge_result.reasoning,
            "Match?": ground_truth == judge_result.label,
        })

    return pd.DataFrame(rows)


def summarize(model_name: str, results: pd.DataFrame):
    total = len(results)
    agreement = results["Match?"].mean()

    print(f"\n=== Summary for {model_name} ===")
    print(f"Rows to judge: {total}")
    print("Judge predictions -- Failure Category breakdown:")
    for cat, n in results["Failure Category"].value_counts().reindex(FAILURE_CATEGORIES, fill_value=0).items():
        print(f"  {cat:>22}: {n}")
    print(f"Agreement (Judge Prediction matches Ground Truth): {agreement:.2%}")


# --------------------------------------------------------------------------
# 4. Main: run for every model, write a CSV each, then compare side by side
# --------------------------------------------------------------------------
def main():
    _set_api_key()

    df = pd.read_csv(INPUT_CSV)

    all_results = {}
    for model_name in MODELS_TO_RUN:
        print(f"\nRunning LLM-as-judge with model: {model_name}", file=sys.stderr)
        results = run_judge_for_model(model_name, df)
        out_path = f"LLM-J-{model_name}.csv"
        results.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")
        summarize(model_name, results)
        all_results[model_name] = results

    print("\n\n=== Side-by-side comparison ===")
    compare_cols = ["Test ID", "Ground Truth Intent", "Predicted Intent"]
    comparison = all_results[MODELS_TO_RUN[0]][compare_cols].copy()
    for model_name in MODELS_TO_RUN:
        r = all_results[model_name]
        comparison[f"{model_name}: Failure Category"] = r["Failure Category"].values
        comparison[f"{model_name}: Judge Prediction"] = r["Judge Prediction"].values
        comparison[f"{model_name}: Match?"] = r["Match?"].values

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(comparison.to_string(index=False))

    print("\n=== Agreement rate by model ===")
    for model_name in MODELS_TO_RUN:
        print(f"  {model_name:>15}: {all_results[model_name]['Match?'].mean():.2%}")

    models_agree = all_results[MODELS_TO_RUN[0]]["Judge Prediction"].values == all_results[MODELS_TO_RUN[1]]["Judge Prediction"].values
    print(f"\n{MODELS_TO_RUN[0]} vs {MODELS_TO_RUN[1]} -- Judge Prediction agreement with each other: {models_agree.mean():.2%}")

    comparison_path = "LLM-J-comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"\nWrote {comparison_path}")


if __name__ == "__main__":
    main()
