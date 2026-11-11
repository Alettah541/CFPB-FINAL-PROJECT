"""Draw the held-out set of 30 complaints for blind human labeling.

The 30 development complaints (data/processed/cases.csv) were used to choose
Prompt C, so its accuracy on them is somewhat optimistic. This script draws 30
new complaints that no prompt has seen, for a clean re-test of Prompts A/B/C
and for independent, blind classification by both raters.

Sampling design: stratified by category, at least 3 per category (24), with
the remaining 6 allocated by each category's share of the full file (largest
remainder). This over-represents small categories, so raw accuracy on this set
is not a real-world estimate; reweight by full-file category shares.
Development-set complaints are excluded. Narratives are cut to 2,000
characters, as in Week 7.

Writes to data/processed/holdout/:
  holdout_cases.csv          prompt input: ID + narrative, no labels
                             (30 complaints + the Week 7 adversarial case)
  holdout_answer_key.csv     neutral label ID -> complaint ID, CFPB issue/category
                             (raters: don't open until both sheets are done)
  labeling_sheet_<rater>.xlsx  blank sheet per rater: H01-H30, narrative,
                             category dropdown, note; definitions + instructions tabs

Never overwrites: if the files exist, it checks that a fresh draw reproduces them.

Run from anywhere:  python src/draw_holdout.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = ROOT / "data/raw/CFPB_credit_card_complaints_with_narratives_2015_2017.xlsx"
DEV_CASES = ROOT / "data/processed/cases.csv"
HOLDOUT_DIR = ROOT / "data/processed/holdout"

HOLDOUT_SEED = 4340
N_HOLDOUT = 30
MIN_PER_CATEGORY = 3
MAX_CHARS = 2000  # same cut as the Week 7 sample
RATERS = ["hunter", "alettah"]

# Same mapping as the notebook (cell 18): 30 CFPB issues -> 8 categories
CATEGORY_ISSUES = {
    "billing_dispute": ["Billing disputes", "Billing statement", "Transaction issue"],
    "fees_interest": [
        "APR or interest rate", "Late fee", "Other fee",
        "Balance transfer fee", "Cash advance fee", "Overlimit fee",
    ],
    "fraud_identity": [
        "Identity theft / Fraud / Embezzlement",
        "Unsolicited issuance of credit card", "Privacy",
    ],
    "account_management": [
        "Closing/Cancelling account", "Credit line increase/decrease",
        "Credit determination", "Application processing delay",
        "Payoff process", "Sale of account", "Balance transfer",
        "Cash advance", "Convenience checks",
    ],
    "collections_hardship": [
        "Delinquent account", "Bankruptcy",
        "Forbearance / Workout plans", "Arbitration",
    ],
    "customer_service": ["Customer service / Customer relations"],
    "rewards_marketing": [
        "Rewards", "Advertising and marketing",
        "Credit card protection / Debt protection",
    ],
    "other": ["Other"],
}

# Same category definitions given to the model in Prompts B and C
CATEGORY_DEFINITIONS = {
    "billing_dispute": "disputed charges, statement errors, transactions not posted or posted wrong",
    "fees_interest": "APR or interest rate changes, late fees, other fees",
    "fraud_identity": "identity theft, unauthorized accounts or charges, privacy",
    "account_management": "closing accounts, credit line changes, credit decisions, payoff, balance transfers",
    "collections_hardship": "delinquency, collections, bankruptcy, workout plans, arbitration",
    "customer_service": "how the customer was treated by representatives or support",
    "rewards_marketing": "rewards points, advertising, add-on protection products",
    "other": "fits none of the categories above",
}

INSTRUCTIONS = [
    "Pick ONE category per complaint from the dropdown.",
    "If a complaint fits two categories, pick the primary one and write 'ambiguous: <second category>' in the note.",
    "Use other only if a complaint fits none of the other categories.",
    "Text shown as XXXX is redacted; don't guess what it says.",
    "Label on your own: don't look at the answer key, the AI output, or your partner's sheet until both sheets are done.",
]


def load_complaints():
    df = pd.read_excel(RAW_DATA)
    issue_to_category = {i: c for c, issues in CATEGORY_ISSUES.items() for i in issues}
    df["category"] = df["Issue"].map(issue_to_category)
    assert df["category"].notna().all(), "Unmapped CFPB issue"
    return df


def draw(df, dev_cases):
    dev_ids = set(dev_cases.loc[dev_cases["case_type"] == "typical", "complaint_id"].astype(str))
    pool = df[~df["Complaint ID"].astype(str).isin(dev_ids)]
    assert len(pool) == len(df) - len(dev_ids)

    # Allocation: 3 per category, the remaining 6 by full-file share (largest remainder)
    share = df["category"].value_counts(normalize=True)
    extra = share * (N_HOLDOUT - MIN_PER_CATEGORY * len(share))
    alloc = MIN_PER_CATEGORY + np.floor(extra).astype(int)
    leftover = N_HOLDOUT - alloc.sum()
    alloc[(extra - np.floor(extra)).sort_values(ascending=False).index[:leftover]] += 1

    assert alloc.sum() == N_HOLDOUT
    assert set(alloc.index) == set(CATEGORY_ISSUES)
    assert (alloc >= MIN_PER_CATEGORY).all()

    sample = pd.concat([
        pool[pool["category"] == category].sample(n=n, random_state=HOLDOUT_SEED)
        for category, n in alloc.items()
    ])
    holdout = pd.DataFrame({
        "complaint_id": sample["Complaint ID"].astype(str),
        "date_received": pd.to_datetime(sample["Date received"]).dt.date,
        "issue": sample["Issue"],
        "category": sample["category"],
        "narrative": sample["Consumer complaint narrative"].str.slice(0, MAX_CHARS),
    })
    assert holdout["complaint_id"].is_unique
    assert not set(holdout["complaint_id"]) & dev_ids

    # Shuffle, then assign neutral IDs for the labeling sheets
    holdout = holdout.sample(frac=1, random_state=HOLDOUT_SEED).reset_index(drop=True)
    holdout["label_id"] = [f"H{i:02d}" for i in range(1, N_HOLDOUT + 1)]

    # Prompt input: narratives only, plus the same planted adversarial complaint as Week 7
    adversarial = dev_cases.loc[dev_cases["case_type"] == "adversarial",
                                ["complaint_id", "narrative", "case_type"]]
    holdout_cases = pd.concat([
        holdout[["complaint_id", "narrative"]].assign(case_type="typical"),
        adversarial.astype({"complaint_id": str}),
    ]).sample(frac=1, random_state=HOLDOUT_SEED).reset_index(drop=True)
    assert len(holdout_cases) == N_HOLDOUT + 1

    summary = pd.DataFrame({
        "holdout_n": alloc,
        "holdout_%": (alloc / N_HOLDOUT * 100).round(1),
        "full_file_%": (share * 100).round(1),
    })
    return holdout, holdout_cases, summary


def write_labeling_sheet(path, sheet):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sheet.to_excel(writer, sheet_name="labels", index=False)
        pd.DataFrame(list(CATEGORY_DEFINITIONS.items()),
                     columns=["category", "definition"]).to_excel(
            writer, sheet_name="definitions", index=False)
        pd.DataFrame({"instructions": INSTRUCTIONS}).to_excel(
            writer, sheet_name="instructions", index=False)

    wb = load_workbook(path)
    ws = wb["labels"]
    for col, width in zip("ABCD", [10, 100, 24, 40]):
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.freeze_panes = "A2"

    # Category dropdown so labels can't be misspelled
    dropdown = DataValidation(type="list", allow_blank=True,
                              formula1='"' + ",".join(CATEGORY_DEFINITIONS) + '"')
    ws.add_data_validation(dropdown)
    dropdown.add(f"C2:C{N_HOLDOUT + 1}")

    wb["definitions"].column_dimensions["A"].width = 24
    wb["definitions"].column_dimensions["B"].width = 90
    wb["instructions"].column_dimensions["A"].width = 110
    wb.save(path)


def main():
    assert set(CATEGORY_DEFINITIONS) == set(CATEGORY_ISSUES)
    df = load_complaints()
    dev_cases = pd.read_csv(DEV_CASES, dtype={"complaint_id": str})
    holdout, holdout_cases, summary = draw(df, dev_cases)

    answer_key = holdout[["label_id", "complaint_id", "date_received", "issue", "category"]]
    sheet = holdout[["label_id", "narrative"]].assign(category="", note="")

    print(summary.to_string())
    print(f"\nNarratives cut to {MAX_CHARS:,} characters: "
          f"{(holdout['narrative'].str.len() == MAX_CHARS).sum()} of {N_HOLDOUT}\n")

    paths = {
        "cases": HOLDOUT_DIR / "holdout_cases.csv",
        "key": HOLDOUT_DIR / "holdout_answer_key.csv",
        **{r: HOLDOUT_DIR / f"labeling_sheet_{r}.xlsx" for r in RATERS},
    }

    if all(p.exists() for p in paths.values()):
        saved_cases = pd.read_csv(paths["cases"], dtype={"complaint_id": str})
        saved_key = pd.read_csv(paths["key"], dtype={"complaint_id": str})
        assert saved_cases["complaint_id"].tolist() == holdout_cases["complaint_id"].tolist()
        assert saved_key["complaint_id"].tolist() == answer_key["complaint_id"].tolist()
        print("Holdout files already exist; fresh draw matches them. Nothing overwritten.")
        return

    assert not any(p.exists() for p in paths.values()), (
        "Some holdout files exist but not all; check data/processed/holdout/ by hand")
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    holdout_cases.to_csv(paths["cases"], index=False)
    answer_key.to_csv(paths["key"], index=False)
    for rater in RATERS:
        write_labeling_sheet(paths[rater], sheet)
    print("Wrote:")
    for p in paths.values():
        print("  ", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
