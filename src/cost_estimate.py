"""Estimate the API cost of the held-out runs for Prompts A, B and C.

Reads the token counts saved in data/processed/holdout/generation_results.json and
applies the published Claude Haiku 5.5 rates for prompts under 100,000 tokens.

Rates are an ASSUMPTION (see README, "Cost assumption"): $0.10 per million input tokens
and $0.50 per million output tokens, taken from third-party pricing summaries. Confirm
them against Anthropic's pricing page before publishing a cost figure.

Run from anywhere: python src/cost_estimate.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "data" / "processed" / "holdout" / "generation_results.json"

INPUT_PER_MILLION = 0.10   # USD per 1M input tokens (assumed; see README)
OUTPUT_PER_MILLION = 0.50  # USD per 1M output tokens (assumed; see README)
COMPLAINTS_PER_RUN = 31    # 30 held-out complaints plus 1 adversarial


def run_cost(input_tokens, output_tokens):
    return (input_tokens * INPUT_PER_MILLION + output_tokens * OUTPUT_PER_MILLION) / 1_000_000


def main():
    gen = json.loads(GEN.read_text(encoding="utf8"))
    rows = []
    for prompt in ["A", "B", "C"]:
        i, o = gen[prompt]["input_tokens"], gen[prompt]["output_tokens"]
        cost = run_cost(i, o)
        rows.append((prompt, i, o, i + o, cost, cost / COMPLAINTS_PER_RUN * 1000))
    print(f"{'Prompt':<7}{'Input':>8}{'Output':>9}{'Total':>9}{'Per run':>10}{'Per 1,000':>12}")
    for p, i, o, t, c, per_k in rows:
        print(f"{p:<7}{i:>8,}{o:>9,}{t:>9,}{'$%.4f' % c:>10}{'$%.3f' % per_k:>12}")
    return rows


if __name__ == "__main__":
    main()
