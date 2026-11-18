"""Bootstrap 95% CI for macro F1 on the held-out set, for Prompts A, B and C.

Same method as the accuracy intervals in final_project.ipynb: resample the 30 complaints
with replacement, 1,000 times, with seed 42, and take the 2.5th and 97.5th percentiles.
Uses the saved predictions and CFPB labels only; no API calls.
Run from anywhere.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = ROOT / "data" / "processed" / "holdout"

cases = pd.read_csv(HOLDOUT / "holdout_cases.csv", dtype={"complaint_id": str})
key = pd.read_csv(HOLDOUT / "holdout_answer_key.csv", dtype={"complaint_id": str})
gen = json.loads((HOLDOUT / "generation_results.json").read_text(encoding="utf8"))
real = cases[cases["case_type"] == "typical"].merge(
    key[["complaint_id", "category"]], on="complaint_id", validate="one_to_one")
CATEGORIES = sorted(key["category"].unique())


def macro_f1(true, pred):
    return f1_score(true, pred, labels=CATEGORIES, average="macro", zero_division=0)


def main():
    rows = []
    for prompt in ["A", "B", "C"]:
        preds = pd.DataFrame(gen[prompt]["classifications"])
        preds["complaint_id"] = preds["id"].astype(str)
        ev = real.merge(preds[["complaint_id", "category"]].rename(columns={"category": "predicted"}),
                        on="complaint_id", validate="one_to_one")
        true = ev["category"].to_numpy()
        pred = ev["predicted"].to_numpy()
        point = macro_f1(true, pred)
        rng = np.random.default_rng(42)
        boot = []
        for _ in range(1000):
            idx = rng.integers(0, len(true), len(true))
            boot.append(macro_f1(true[idx], pred[idx]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append((prompt, point, lo, hi))
        print(f"Prompt {prompt}: macro F1 {point:.3f}, 95% bootstrap CI {lo:.3f} to {hi:.3f}")
    return rows


if __name__ == "__main__":
    main()
