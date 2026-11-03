"""Stage 3: For each top-k candidate pair (r1, r2), compute two diagnostics
restricted to instances of r1 ∪ r2:

  (a) Classification accuracy of a "majority class per (h_upos, d_upos) cell"
      classifier predicting relation in {r1, r2}. If acc > 0.95, the
      distinction is essentially deterministic given POS.

  (b) Conditional entropy H(R | head_upos, dep_upos). If near 0, ditto.

  H(R | H, D) = sum_{h,d} P(h,d) * H(R | h, d)
             = sum_{h,d} (n_{h,d} / N) * binary_entropy(p_{r1 | h,d})

Usage:
    python 03_pair_analysis.py <train.conllu> [tvd.csv] [output_dir] [top_k]
"""
import os
import sys
from collections import Counter, defaultdict
from math import log2

import pandas as pd

from utils import parse_conllu, base_relation, get_head_upos, should_skip


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * log2(p) - (1 - p) * log2(1 - p)


def analyse_pair(train_path: str, r1: str, r2: str):
    """Return (accuracy, cond_entropy, total_count, cell_breakdown_rows)."""
    cell_counts = defaultdict(Counter)  # (h, d) -> Counter[r1, r2]
    total = 0
    for sent in parse_conllu(train_path):
        for tok in sent:
            if should_skip(tok, sent):
                continue
            rel = base_relation(tok["deprel"])
            if rel != r1 and rel != r2:
                continue
            h_upos = get_head_upos(sent, tok)
            d_upos = tok["upos"]
            cell_counts[(h_upos, d_upos)][rel] += 1
            total += 1
    if total == 0:
        return None

    correct = 0
    H_cond = 0.0
    breakdown = []
    for cell, counts in sorted(cell_counts.items(), key=lambda x: -sum(x[1].values())):
        n_cell = sum(counts.values())
        n_r1 = counts.get(r1, 0)
        n_r2 = counts.get(r2, 0)
        majority = max(n_r1, n_r2)
        correct += majority
        p1 = n_r1 / n_cell
        H_cell = binary_entropy(p1)
        H_cond += (n_cell / total) * H_cell
        breakdown.append(
            {
                "head_upos": cell[0],
                "dep_upos": cell[1],
                "n_total": n_cell,
                f"n_{r1}": n_r1,
                f"n_{r2}": n_r2,
                "majority": r1 if n_r1 >= n_r2 else r2,
                "cell_entropy": H_cell,
            }
        )
    return correct / total, H_cond, total, breakdown


def main(train_path, tvd_csv, output_dir, top_k=20):
    tvd_df = pd.read_csv(tvd_csv).head(top_k)

    summary_rows = []
    breakdowns_dir = os.path.join(output_dir, "03_pair_breakdowns")
    os.makedirs(breakdowns_dir, exist_ok=True)

    for _, row in tvd_df.iterrows():
        r1, r2 = row["rel1"], row["rel2"]
        out = analyse_pair(train_path, r1, r2)
        if out is None:
            continue
        acc, H, n, breakdown = out
        deterministic = (acc > 0.95) or (H < 0.10)
        summary_rows.append(
            {
                "rel1": r1,
                "rel2": r2,
                "tvd": row["tvd"],
                "count1": row["count1"],
                "count2": row["count2"],
                "n_total": n,
                "majority_acc": acc,
                "cond_entropy": H,
                "deterministic_pos": deterministic,
            }
        )
        bd_df = pd.DataFrame(breakdown)
        bd_df.to_csv(os.path.join(breakdowns_dir, f"{r1}_vs_{r2}.csv"), index=False)

    out_df = pd.DataFrame(summary_rows)
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "03_pair_analysis.csv")
    out_df.to_csv(out_csv, index=False)

    out_txt = os.path.join(output_dir, "03_deterministic_pairs.txt")
    with open(out_txt, "w") as f:
        f.write(f"Top {top_k} candidate pairs by TVD, with diagnostics.\n")
        f.write(
            "deterministic_pos = (majority_acc > 0.95) OR (cond_entropy < 0.10)\n\n"
        )
        f.write(out_df.to_string(index=False))
        f.write("\n\nDeterministic-given-POS pairs:\n")
        det = out_df[out_df["deterministic_pos"]]
        f.write(det.to_string(index=False))
        f.write("\n")

    # Suggested collapse rules from deterministic-given-POS pairs.
    #
    # A pair has to pass three filters to be a candidate:
    #
    #   1. deterministic_pos = True  (acc > 0.95 or H < 0.10)
    #   2. tvd <= TVD_MAX            (they share POS cells - pairs at
    #                                 TVD near 1 have only accidental overlap)
    #   3. SHARED_CELL_MIN cells where BOTH relations have >= MIN_PER_CELL
    #      tokens   (real overlap, not three edge-case cells)
    #
    # Direction defaults to: less frequent -> more frequent. Sorted by TVD
    # ascending so the candidates with the most distributional overlap (i.e.
    # the most "same population cleanly split by POS" signal) come first.
    #
    # No automatic dedup: linguistic judgement is required to pick a subset.
    # iobj/nsubj will pass these filters and is not a sensible collapse;
    # obl/nmod will pass these filters and IS sensible. The pipeline cannot
    # tell the difference - that's your job. Review and prune before using.
    TVD_MAX = 0.99
    SHARED_CELL_MIN = 2
    MIN_PER_CELL = 3

    def shared_cell_count(r1: str, r2: str) -> int:
        bd = pd.read_csv(os.path.join(breakdowns_dir, f"{r1}_vs_{r2}.csv"))
        col1, col2 = f"n_{r1}", f"n_{r2}"
        return int(((bd[col1] >= MIN_PER_CELL) & (bd[col2] >= MIN_PER_CELL)).sum())

    suggest_rows = []
    for _, r in out_df.iterrows():
        if not r["deterministic_pos"]:
            continue
        if r["tvd"] > TVD_MAX:
            continue
        try:
            nshared = shared_cell_count(r["rel1"], r["rel2"])
        except FileNotFoundError:
            continue
        if nshared < SHARED_CELL_MIN:
            continue
        suggest_rows.append(
            {
                "rel1": r["rel1"],
                "rel2": r["rel2"],
                "tvd": r["tvd"],
                "majority_acc": r["majority_acc"],
                "cond_entropy": r["cond_entropy"],
                "count1": r["count1"],
                "count2": r["count2"],
                "n_shared_cells": nshared,
            }
        )
    suggest = pd.DataFrame(suggest_rows).sort_values("tvd", ascending=True)

    sug_path = os.path.join(output_dir, "03_suggested_rules.txt")
    with open(sug_path, "w") as f:
        f.write(
            "# Candidate collapse rules from deterministic-given-POS pairs.\n"
            "# Format:  source=target   # tvd / acc / entropy / n_shared_cells\n"
            f"# Filters: deterministic_pos=True, tvd <= {TVD_MAX},\n"
            f"#          shared cells (>= {MIN_PER_CELL} tokens of each) >= {SHARED_CELL_MIN}\n"
            "# Direction: less frequent -> more frequent. Sorted by TVD ascending.\n"
            "# THIS IS A CANDIDATE LIST, NOT A RECOMMENDATION. Statistical\n"
            "# similarity is not the same as linguistic redundancy. iobj/nsubj\n"
            "# will pass; collapsing them is wrong. obl/nmod will pass; collapsing\n"
            "# them is fine. Review and delete the lines you do not want before\n"
            "# passing the rest to stage 4 / pipeline as --rules src=tgt,src2=tgt2.\n"
            "# Avoid chains: do not keep both 'a=b' and 'b=c' in the final list.\n\n"
        )
        for _, r in suggest.iterrows():
            r1, r2, c1, c2 = r["rel1"], r["rel2"], r["count1"], r["count2"]
            src, tgt = (r2, r1) if c1 >= c2 else (r1, r2)
            f.write(
                f"{src}={tgt}    # tvd={r['tvd']:.4f} acc={r['majority_acc']:.4f} "
                f"H={r['cond_entropy']:.4f} shared={r['n_shared_cells']}\n"
            )

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_txt}")
    print(f"Wrote {sug_path}")
    print(f"Wrote per-pair breakdowns into {breakdowns_dir}/")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    train_path = sys.argv[1] if len(sys.argv) > 1 else "train.conllu"
    tvd_csv = sys.argv[2] if len(sys.argv) > 2 else "outputs/02_tvd_ranking.csv"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "outputs"
    top_k = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    main(train_path, tvd_csv, output_dir, top_k)