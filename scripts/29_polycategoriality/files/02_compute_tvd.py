"""Stage 2: For every pair of relations, treat each row of the count table
as a probability distribution over (head_upos, dep_upos) pairs, and compute
the total variation distance (TVD).

  TVD(P, Q) = 0.5 * sum_x |P(x) - Q(x)|   in [0, 1]

TVD = 1 means the two distributions are disjoint (no pair occurs under both),
TVD = 0 means identical distributions. High TVD => candidate for "the
distinction is essentially POS-determined".

Usage:
    python 02_compute_tvd.py [table.csv] [output_dir] [min_count]
"""
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd


def compute_tvd(table_csv: str, output_dir: str, min_count: int = 100):
    df = pd.read_csv(table_csv)
    df = df[df["total"] >= min_count].reset_index(drop=True)

    rels = df["relation"].tolist()
    pair_cols = [c for c in df.columns if c not in ("relation", "total")]

    M = df[pair_cols].to_numpy(dtype=float)
    row_sums = M.sum(axis=1, keepdims=True)
    P = M / row_sums  # row-normalised: P[i] is P_{r_i} over (h,d) pairs

    rows = []
    for i, j in combinations(range(len(rels)), 2):
        tvd = 0.5 * np.abs(P[i] - P[j]).sum()
        rows.append(
            {
                "rel1": rels[i],
                "rel2": rels[j],
                "count1": int(df["total"][i]),
                "count2": int(df["total"][j]),
                "tvd": tvd,
            }
        )

    out = pd.DataFrame(rows).sort_values("tvd", ascending=False).reset_index(drop=True)
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "02_tvd_ranking.csv")
    out.to_csv(out_csv, index=False)

    out_txt = os.path.join(output_dir, "02_tvd_top.txt")
    with open(out_txt, "w") as f:
        f.write(f"min_count filter: {min_count}\n")
        f.write(f"relations kept: {len(rels)}\n")
        f.write(f"pairs scored:   {len(rows)}\n\n")
        f.write("Top 30 most disjoint relation pairs (highest TVD)\n")
        f.write("rel1, rel2, count1, count2, tvd\n")
        for _, r in out.head(30).iterrows():
            f.write(
                f"  {r.rel1:12s} {r.rel2:12s} {r.count1:6d} {r.count2:6d} {r.tvd:.4f}\n"
            )

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_txt}")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    table_csv = sys.argv[1] if len(sys.argv) > 1 else "outputs/01_relation_pos_table.csv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs"
    min_count = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    compute_tvd(table_csv, output_dir, min_count)
