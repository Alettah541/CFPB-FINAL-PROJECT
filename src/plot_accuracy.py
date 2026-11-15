"""Accuracy by prompt with bootstrap 95% CIs, saved to outputs/accuracy_by_prompt.png.

Same method as final_project.ipynb cells 33 and 34: 1,000 bootstrap resamples, seed 42.
Run from the notebooks/ folder or the repo root.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = ROOT / "data" / "processed" / "holdout"
OUT = ROOT / "outputs" / "accuracy_by_prompt.png"

cases = pd.read_csv(HOLDOUT / "holdout_cases.csv", dtype={"complaint_id": str})
key = pd.read_csv(HOLDOUT / "holdout_answer_key.csv", dtype={"complaint_id": str})
gen = json.loads((HOLDOUT / "generation_results.json").read_text(encoding="utf8"))

real = cases[cases["case_type"] == "typical"].merge(
    key[["complaint_id", "category"]], on="complaint_id", validate="one_to_one")
assert len(real) == 30

results = {}
for prompt in ["A", "B", "C"]:
    preds = pd.DataFrame(gen[prompt]["classifications"])
    preds["id"] = preds["id"].astype(str)
    cmp_ = real.merge(preds[["id", "category"]], left_on="complaint_id", right_on="id",
                      suffixes=("_actual", "_predicted"), validate="one_to_one")
    correct = (cmp_["category_actual"] == cmp_["category_predicted"]).astype(int).to_numpy()
    rng = np.random.default_rng(42)
    boot = [rng.choice(correct, size=len(correct), replace=True).mean() for _ in range(1000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    results[prompt] = (correct.mean(), lo, hi)
    print(f"Prompt {prompt}: accuracy {correct.mean():.1%}, 95% CI {lo:.1%} to {hi:.1%}")

# Check against the notebook's printed intervals before saving the chart
assert round(results["A"][1] * 100, 1) == 33.2 and round(results["A"][2] * 100, 1) == 70.0
assert round(results["B"][1] * 100, 1) == 33.3 and round(results["B"][2] * 100, 1) == 66.7
assert round(results["C"][1] * 100, 1) == 33.3 and round(results["C"][2] * 100, 1) == 66.7

prompts = list(results)
acc = [results[p][0] * 100 for p in prompts]
lo_err = [results[p][0] * 100 - results[p][1] * 100 for p in prompts]
hi_err = [results[p][2] * 100 - results[p][0] * 100 for p in prompts]

fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
bars = ax.bar(prompts, acc, color="#44546A", width=0.55)
ax.errorbar(prompts, acc, yerr=[lo_err, hi_err], fmt="none", ecolor="black", capsize=6, linewidth=1.2)
ax.axhline(13.3, color="#8a0f2b", linestyle="--", linewidth=1)
ax.text(2.45, 14.5, "always-billing baseline 13.3%", color="#8a0f2b", ha="right", fontsize=8)
for b, a in zip(bars, acc):
    ax.text(b.get_x() + b.get_width() / 2, 3, f"{a:.0f}%", ha="center", color="white", fontweight="bold")
ax.set_ylim(0, 100)
ax.set_ylabel("Accuracy vs CFPB label (%)")
ax.set_title("Held-out accuracy by prompt (n = 30, 95% bootstrap CI)", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT)
print("saved", OUT)
