"""Bootstrap 95% CIs for the kappas and the Prompt C minus Prompt A brief score gap.

Same settings as src/macro_f1_ci.py: 1,000 resamples, seed 42, 2.5th and 97.5th
percentiles. Uses the saved labeling sheets, human score files and judge results only;
no API calls. Run from anywhere.

What is resampled (the units are small, so read these intervals as rough):
  - Blind category kappa: the 30 held-out complaints.
  - Rubric kappa (rater vs rater) and judge-vs-human kappa: the 15 brief x dimension
    scores. These are not fully independent (5 dimensions per brief), so the intervals
    are probably too narrow.
  - C minus A human brief gap: the 10 rater x dimension paired differences (C minus A).
    The gap is 5 x the mean paired difference (the mean of two raters' totals).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / "data" / "processed" / "holdout"
DIMENSIONS = ["grounding", "recurring_issues", "actionability", "clarity", "transparency"]
SCALE = [0, 1, 2, 3, 4]
N_BOOT = 1000


def ci(values):
    values = np.array([v for v in values if not np.isnan(v)])
    return np.percentile(values, [2.5, 97.5]), len(values)


def main():
    rng = np.random.default_rng(42)
    key = pd.read_csv(H / "holdout_answer_key.csv", dtype={"complaint_id": str})
    cats = sorted(key["category"].unique())

    # Blind category kappa (30 complaints)
    sheets = {r: pd.read_excel(H / f"labeling_sheet_{r}.xlsx", sheet_name="labels")
              for r in ["hunter", "alettah"]}
    h = sheets["hunter"]["category"].astype("string").str.strip().to_numpy()
    a = sheets["alettah"]["category"].astype("string").str.strip().to_numpy()
    point = cohen_kappa_score(h, a, labels=cats)
    boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(h), len(h))
        boot.append(cohen_kappa_score(h[i], a[i], labels=cats))
    (lo, hi), n = ci(boot)
    print(f"Blind category kappa {point:.3f}, 95% CI {lo:.3f} to {hi:.3f} ({n} usable resamples)")

    # Human scores, one row per brief x dimension, labeled with the prompt (A, B, C)
    brief_key = pd.read_csv(H / "brief_key.csv").set_index("brief")["output"]
    scores = {}
    for r in ["hunter", "alettah"]:
        df = pd.read_csv(H / f"human_scores_{r}.csv")
        df["output"] = df["brief"].map(brief_key)
        scores[r] = df.set_index(["output", "dimension"])["score"]
    idx = [(o, d) for o in "ABC" for d in DIMENSIONS]
    hs = scores["hunter"].loc[idx].to_numpy()
    al = scores["alettah"].loc[idx].to_numpy()

    def k(x, y):
        return cohen_kappa_score(x, y, labels=SCALE, weights="linear")

    point = k(hs, al)
    boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(idx), len(idx))
        boot.append(k(hs[i], al[i]))
    (lo, hi), n = ci(boot)
    print(f"Rubric kappa (human vs human) {point:.3f}, 95% CI {lo:.3f} to {hi:.3f} ({n} usable resamples)")

    # Judge vs human: mean of the two raters' kappas against the judge's eval run
    judge = json.loads((H / "judge_results.json").read_text(encoding="utf8"))
    jd = np.array([judge[f"{o}_eval"][d] for o, d in idx])
    point = np.mean([k(hs, jd), k(al, jd)])
    boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(idx), len(idx))
        boot.append(np.mean([k(hs[i], jd[i]), k(al[i], jd[i])]))
    (lo, hi), n = ci(boot)
    print(f"Judge vs human kappa (mean of two) {point:.3f}, 95% CI {lo:.3f} to {hi:.3f} ({n} usable resamples)")

    # C minus A gap in human brief totals
    diffs = np.array([scores[r].loc[("C", d)] - scores[r].loc[("A", d)]
                      for r in ["hunter", "alettah"] for d in DIMENSIONS])
    gap = 5 * diffs.mean()
    boot = [5 * diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(N_BOOT)]
    (lo, hi), n = ci(boot)
    per_rater = {r: int(scores[r].loc["C"].sum() - scores[r].loc["A"].sum()) for r in scores}
    print(f"C minus A brief gap {gap:.2f}, 95% CI {lo:.2f} to {hi:.2f}; per rater {per_rater}")


if __name__ == "__main__":
    main()
