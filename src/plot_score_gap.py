"""Gap between brief scores vs the MDD (minimum detectable difference).

Bars: the C-minus-A and A-minus-B gaps in human brief totals, plus the average
disagreement between the two raters. The dashed line is the MDD.
Saved to outputs/score_gap.png. Run from anywhere.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from agreement import mdd_from

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = ROOT / "data" / "processed" / "holdout"
OUT = ROOT / "outputs" / "score_gap.png"

hs = pd.concat([pd.read_csv(HOLDOUT / "human_scores_hunter.csv"),
                pd.read_csv(HOLDOUT / "human_scores_alettah.csv")])
bk = pd.read_csv(HOLDOUT / "brief_key.csv")
hs["output"] = hs["brief"].map(dict(zip(bk["brief"], bk["output"])))
human = hs.groupby(["output", "rater"])["score"].sum().unstack()
raters = list(human.columns)

# Rater disagreement per brief, and the MDD as in final_project.ipynb cell 48
disagreements = [abs(human.loc[p, raters[0]] - human.loc[p, raters[1]]) for p in human.index]
mdd = mdd_from(disagreements)
mean_disagreement = sum(disagreements) / len(disagreements)

means = human.mean(axis=1)
gap_c_a = means["C"] - means["A"]
gap_a_b = means["A"] - means["B"]

labels = ["C minus A\n(brief gap)", "A minus B\n(brief gap)", "Average rater\ndisagreement"]
values = [gap_c_a, gap_a_b, mean_disagreement]
colors = ["#B5532A", "#8A9AA8", "#8A9AA8"]

fig, ax = plt.subplots(figsize=(7, 4.4), dpi=150)
bars = ax.bar(labels, values, color=colors, width=0.55)
ax.axhline(mdd, color="#1D5C7A", linestyle="--", linewidth=1.5)
ax.text(2.35, mdd + 0.12, f"MDD = {mdd:.2f}", color="#1D5C7A", fontsize=9, ha="right")
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v:.1f}", ha="center", fontsize=10, color="#1B2430")
ax.set_ylabel("Points on the 20-point brief score")
ax.set_ylim(0, max(values) + 1.2)
ax.set_title("Only the C gap is above the MDD", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT)

print(f"MDD {mdd:.2f}; C-A {gap_c_a:.1f}; A-B {gap_a_b:.1f}; mean disagreement {mean_disagreement:.2f}")
print("saved", OUT)
