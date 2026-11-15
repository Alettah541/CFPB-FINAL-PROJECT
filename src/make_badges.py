"""Write shields.io endpoint JSON for the README badges, computed from project data.

Writes outputs/badges/*.json. The README badges read these files from GitHub, so
run this script and push the JSON whenever the numbers change.
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from agreement import classification_kappa, mdd_from

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = ROOT / "data" / "processed" / "holdout"
OUT = ROOT / "outputs" / "badges"
OUT.mkdir(parents=True, exist_ok=True)

# Held-out accuracy for Prompt C (the same 15 of 30 for all prompts)
cases = pd.read_csv(HOLDOUT / "holdout_cases.csv", dtype={"complaint_id": str})
key = pd.read_csv(HOLDOUT / "holdout_answer_key.csv", dtype={"complaint_id": str})
gen = json.loads((HOLDOUT / "generation_results.json").read_text(encoding="utf8"))
real = cases[cases["case_type"] == "typical"].merge(
    key[["complaint_id", "category"]], on="complaint_id", validate="one_to_one")
preds = pd.DataFrame(gen["C"]["classifications"])
preds["complaint_id"] = preds["id"].astype(str)
ev = real.merge(preds[["complaint_id", "category"]].rename(columns={"category": "predicted"}),
                on="complaint_id", validate="one_to_one")
accuracy = (ev["category"] == ev["predicted"]).mean() * 100

# Blind classification kappa from the two labeling sheets
sheets = {r: pd.read_excel(HOLDOUT / f"labeling_sheet_{r}.xlsx", sheet_name="labels")
          for r in ["hunter", "alettah"]}
assert sheets["hunter"]["label_id"].tolist() == sheets["alettah"]["label_id"].tolist()
cats = sorted(key["category"].unique())
kappa = classification_kappa(
    sheets["hunter"]["category"].astype("string").str.strip(),
    sheets["alettah"]["category"].astype("string").str.strip(), cats)

# MDD from the two raters' brief totals
hs = pd.concat([pd.read_csv(HOLDOUT / "human_scores_hunter.csv"),
                pd.read_csv(HOLDOUT / "human_scores_alettah.csv")])
bk = pd.read_csv(HOLDOUT / "brief_key.csv")
hs["output"] = hs["brief"].map(dict(zip(bk["brief"], bk["output"])))
tot = hs.groupby(["output", "rater"])["score"].sum().unstack()
pairs = list(combinations(tot.columns, 2))
diffs = [abs(tot.loc[o, a] - tot.loc[o, b]) for o in tot.index for a, b in pairs]
mdd = mdd_from(diffs)

# Judge gate: the lowest of Prompt C's three judge runs, from judge_results.json
judge = json.loads((HOLDOUT / "judge_results.json").read_text(encoding="utf8"))
gate = min(sum(judge[f"C_run{i}"].values()) for i in (1, 2, 3))


def badge(name, label, message, color):
    (OUT / f"{name}.json").write_text(json.dumps({
        "schemaVersion": 1, "label": label, "message": message, "color": color}, indent=2), encoding="utf8")
    print(f"{name}: {label} = {message}")


# Rubric kappa: linearly weighted, the two raters' 0-4 scores per output and dimension
wide = hs.pivot_table(index=["output", "dimension"], columns="rater", values="score")
rubric_kappa = cohen_kappa_score(wide["hunter"], wide["alettah"], labels=[0, 1, 2, 3, 4], weights="linear")

badge("accuracy", "held-out accuracy", f"{accuracy:.0f}%", "critical")
badge("kappa", "blind kappa", f"{kappa:.3f} (fair)", "orange")
badge("rubric_kappa", "rubric kappa", f"{rubric_kappa:.3f}", "brightgreen")
badge("mdd", "MDD", f"{mdd:.2f}", "informational")
badge("judge_gate", "judge gate", f"{gate}/20", "informational")
