"""Stage 1: build a table where rows are base relation labels (before ':')
and columns are (head_upos, dependent_upos) pairs. Cells are token counts.

Usage:
    python 01_build_table.py <train.conllu> [output_dir]
"""
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

from utils import parse_conllu, base_relation, get_head_upos


def build_table(train_path: str, output_dir: str):
    counts = defaultdict(Counter)  # rel -> Counter[(h_upos, d_upos)]
    rel_totals = Counter()
    pair_totals = Counter()

    n_sents = 0
    n_tokens = 0
    for sent in parse_conllu(train_path):
        n_sents += 1
        for tok in sent:
            n_tokens += 1
            rel = base_relation(tok["deprel"])
            h_upos = get_head_upos(sent, tok)
            d_upos = tok["upos"]
            pair = (h_upos, d_upos)
            counts[rel][pair] += 1
            rel_totals[rel] += 1
            pair_totals[pair] += 1

    rels = sorted(rel_totals.keys())
    pairs = sorted(pair_totals.keys())
    pair_cols = [f"{h}->{d}" for h, d in pairs]

    rows = []
    for rel in rels:
        row = {"relation": rel, "total": rel_totals[rel]}
        for (h, d), col in zip(pairs, pair_cols):
            row[col] = counts[rel].get((h, d), 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "01_relation_pos_table.csv")
    df.to_csv(out_csv, index=False)

    # Also write a human-readable summary file.
    summary_lines = []
    summary_lines.append(f"Train file: {train_path}")
    summary_lines.append(f"Sentences: {n_sents}")
    summary_lines.append(f"Tokens (analysed): {n_tokens}")
    summary_lines.append(f"Distinct base relations: {len(rels)}")
    summary_lines.append(f"Distinct (head_upos, dep_upos) pairs: {len(pairs)}")
    summary_lines.append("")
    summary_lines.append("Relation totals (descending):")
    for rel, n in rel_totals.most_common():
        summary_lines.append(f"  {rel:20s} {n:8d}")
    summary_lines.append("")
    summary_lines.append("Top 20 (head_upos, dep_upos) pairs by count:")
    for pair, n in pair_totals.most_common(20):
        summary_lines.append(f"  {pair[0]:>10s} -> {pair[1]:<10s} {n:8d}")

    out_txt = os.path.join(output_dir, "01_summary.txt")
    with open(out_txt, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_txt}")
    print(f"Relations: {len(rels)} | (h,d) UPOS pairs: {len(pairs)}")


if __name__ == "__main__":
    train_path = sys.argv[1] if len(sys.argv) > 1 else "train.conllu"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs"
    build_table(train_path, output_dir)
