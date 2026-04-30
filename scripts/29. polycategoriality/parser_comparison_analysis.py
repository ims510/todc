"""
purity_entropy_analysis.py
==========================
Secondary analysis on tokens_joined.csv from parser_comparison.py.

Tests two sharp predictions implied by the stability framework:

  P1 — RIVAL ALIGNMENT
       Among POS errors, the parser's pred_upos should equal the
       stability-derived `rival` category significantly more often
       when purity is low — and most strongly when entropy is also
       low (a single clear rival).

  P2 — CONFUSION DIFFUSENESS
       Among POS errors, the entropy of the parser's pred_upos
       distribution should be higher in the high-stability-entropy
       cells than in the low-stability-entropy cells.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy as shannon_entropy

INPUT_CSV  = "./parser_comparison_output/tokens_joined.csv"
OUTPUT_DIR = "./parser_comparison_output"

df = pd.read_csv(INPUT_CSV)

# Keep only tokens with stability scores AND a defined rival
df = df.dropna(subset=["purity", "entropy", "rival"]).copy()
print(f"Scored tokens with rival defined: {len(df)}")

# ---------------------------------------------------------------------------
# Stratify by purity (median split) × entropy (median split)
# ---------------------------------------------------------------------------
pur_med = df["purity"].median()
ent_med = df["entropy"].median()
print(f"Purity median  = {pur_med:.3f}")
print(f"Entropy median = {ent_med:.3f}")

df["pur_lvl"] = np.where(df["purity"]  <  pur_med, "low_pur",  "high_pur")
df["ent_lvl"] = np.where(df["entropy"] >= ent_med, "high_ent", "low_ent")
df["cell"]    = df["pur_lvl"] + " × " + df["ent_lvl"]

cells = ["low_pur × low_ent",  "low_pur × high_ent",
         "high_pur × low_ent", "high_pur × high_ent"]

# ---------------------------------------------------------------------------
# P1 — RIVAL ALIGNMENT
# ---------------------------------------------------------------------------
errors = df[df["pos_err"] == 1].copy()
errors["pred_eq_rival"] = (errors["pred_upos"] == errors["rival"]).astype(int)
df["pred_eq_rival"]     = (df["pred_upos"]     == df["rival"]).astype(int)

print("\n=== P1: P(pred_upos == rival | error), by cell ===")
print("Hypothesis: highest in low_pur × low_ent\n")
err_table = (errors.groupby("cell")["pred_eq_rival"]
                   .agg(n_errors="count", rival_hit_rate="mean")
                   .reindex(cells))
print(err_table.to_string())

print("\n=== Baseline: P(pred_upos == rival) across ALL scored tokens ===")
print("If `rival` is a meaningful prediction, error rates above the\n"
      "baseline indicate genuine confusion alignment, not chance.\n")
base_table = (df.groupby("cell")["pred_eq_rival"]
                .agg(n_total="count", baseline_hit_rate="mean")
                .reindex(cells))
print(base_table.to_string())

# Lift over baseline
combined = err_table.join(base_table)
combined["lift"] = combined["rival_hit_rate"] / combined["baseline_hit_rate"]
print("\n=== Lift (error hit rate / baseline hit rate) ===")
print(combined[["n_errors", "rival_hit_rate",
                "baseline_hit_rate", "lift"]].to_string())

# ---------------------------------------------------------------------------
# P2 — CONFUSION DIFFUSENESS
# ---------------------------------------------------------------------------
print("\n=== P2: entropy of parser pred_upos distribution among errors ===")
print("Hypothesis: high_ent cells > low_ent cells\n")

p2_rows = []
for cell in cells:
    cell_errs = errors[errors["cell"] == cell]
    if len(cell_errs) < 5:
        p2_rows.append({"cell": cell, "n_errors": len(cell_errs),
                        "pred_entropy": np.nan, "top_preds": "—"})
        continue
    dist = cell_errs["pred_upos"].value_counts(normalize=True)
    H    = shannon_entropy(dist.values, base=2)
    top3 = ", ".join(f"{k}:{v:.2f}" for k, v in dist.head(3).items())
    p2_rows.append({"cell": cell, "n_errors": len(cell_errs),
                    "pred_entropy": round(H, 3), "top_preds": top3})
print(pd.DataFrame(p2_rows).to_string(index=False))

# ---------------------------------------------------------------------------
# Visualization: rival hit rate by cell, with baseline overlay
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(cells))
err_vals  = combined["rival_hit_rate"].values
base_vals = combined["baseline_hit_rate"].values
ns        = combined["n_errors"].values

w = 0.38
ax.bar(x - w/2, err_vals,  w, label="P(pred=rival | error)", color="steelblue")
ax.bar(x + w/2, base_vals, w, label="P(pred=rival | all)",   color="lightgray")
ax.set_xticks(x)
ax.set_xticklabels(cells, rotation=12, ha="right")
ax.set_ylabel("Probability")
ax.set_title("Parser confusion alignment with stability rival category")
ax.legend()
for xi, n in zip(x, ns):
    ax.text(xi - w/2, err_vals[list(x).index(xi)] + 0.005,
            f"n={n}", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
out = os.path.join(OUTPUT_DIR, "rival_alignment.png")
fig.savefig(out, dpi=150)
print(f"\nSaved: {out}")