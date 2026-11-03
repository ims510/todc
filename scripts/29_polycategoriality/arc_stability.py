"""Arc-stability: the categorial-coherence framework applied to RELATIONS.

Each (lemma, pos) lex unit from the existing tod feature matrix is recategorised
by its DOMINANT incoming relation (the modal `deprel` across all the unit's
tokens in the training conllu). The stability matrix is then computed exactly
as in the user's POS-overlap work, but with relations on the rows/cols.

Reading the output:
  diagonal entry S[r,r]    = self-coherence: how often arcs of relation r
                             cluster with other arcs of relation r
  off-diagonal S[a,b]+S[b,a] = pairwise affinity: high values mean a and b
                             share context space and are candidates for
                             collapse; low values mean they're distinct

This is the natural extension of the POS-overlap work: same machinery, different
categorisation. No simulation, no column-merge — the original matrix is the
diagnostic.

Filtering: relations in EXCLUDE_RELATIONS (punct, root) and lex units with
incoming PUNCT are dropped, matching the rest of the pipeline. Lex units whose
dominant relation accounts for less than --min-dominance of their tokens are
also dropped to avoid noisy categorisation.

Usage:
    python arc_stability.py \\
        --treebank /path/to/UD_French-GSD \\
        --patterns-file /path/to/patterns_all_nodes.txt \\
        --train-conllu /path/to/train.conllu \\
        --output-dir outputs
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# tod setup
import sys
sys.path.insert(1, "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/tod")
import tod.corpus   # noqa: E402


SIM_THRESHOLD = 0.7
MAX_NEIGHBOURS = 20
EXCLUDE_RELATIONS = {"punct", "root"}

EXCLUDED_FEATURE_PATTERNS = [r"own"]
INCLUDED_FEATURE_PATTERNS = [
    r"upos=", r"position=", r"rel_shallow=",
    r"Abbr=", r"Aspect=", r"Animacy=", r"Case=",
    r"Clusivity=", r"Definite=", r"Deixis=", r"DeixisRef",
    r"Evident=", r"Negation=", r"Number=", r"Gender=",
    r"Degree=", r"ExtPos=", r"Foreign=", r"Mood",
    r"NounClass=", r"NumType=", r"Person=", r"Polarity=",
    r"Polite=", r"Poss=", r"PronType=", r"Reflex=",
    r"Tense=", r"Typo=", r"VerbForm=", r"Voice=",
]


def load_corpus(treebank_path, patterns_file):
    return tod.corpus.Corpus(
        treebank_path=treebank_path,
        grew_pattern="pattern{X[upos<>PUNCT]}",
        patterns_text_file=patterns_file,
        use_sud=False,
        matrix_type="coverage",
        excluded_feature_patterns=EXCLUDED_FEATURE_PATTERNS,
        included_feature_patterns=INCLUDED_FEATURE_PATTERNS,
    )


def base_relation(deprel):
    return deprel.split(":", 1)[0]


def get_dominant_relation_per_lexunit(conllu_path, idx2lexunit, min_dominance=0.5):
    """For each (key, pos) unit, count its incoming relations across the corpus
    and return the dominant one if it accounts for at least `min_dominance` of
    the unit's tokens. Auto-detects whether tod uses lemma or form as the first
    field of each lex unit.

    Returns: (lex2relation: dict, stats: dict)
    """
    recognized = {(lu[0], lu[1]) for lu in idx2lexunit.values()}
    counts_lemma = defaultdict(Counter)
    counts_form = defaultdict(Counter)

    with open(conllu_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 10:
                continue
            if "-" in fields[0] or "." in fields[0]:
                continue
            form = fields[1]
            lemma = fields[2]
            upos = fields[3]
            deprel_base = base_relation(fields[7])
            if upos == "PUNCT" or deprel_base in EXCLUDE_RELATIONS:
                continue
            counts_lemma[(lemma, upos)][deprel_base] += 1
            counts_form[(form, upos)][deprel_base] += 1

    n_lemma = sum(1 for k in counts_lemma if k in recognized)
    n_form = sum(1 for k in counts_form if k in recognized)
    print(f"  matching by lemma: {n_lemma} units recognised; "
          f"by form: {n_form} units recognised")
    counts = counts_lemma if n_lemma >= n_form else counts_form

    lex2relation = {}
    n_dropped_dominance = 0
    n_dropped_unrecognised = 0
    for key, ctr in counts.items():
        if key not in recognized:
            n_dropped_unrecognised += 1
            continue
        most_rel, most_count = ctr.most_common(1)[0]
        total = sum(ctr.values())
        if most_count / total < min_dominance:
            n_dropped_dominance += 1
            continue
        lex2relation[key] = most_rel

    return lex2relation, {
        "matched_units": len(lex2relation),
        "dropped_low_dominance": n_dropped_dominance,
        "dropped_unrecognised": n_dropped_unrecognised,
        "matching": "lemma" if n_lemma >= n_form else "form",
    }


def build_stability_matrix(feature_matrix, idx2lexunit, lex2relation,
                           sim_threshold=SIM_THRESHOLD,
                           max_neighbours=MAX_NEIGHBOURS):
    """Stability matrix with relations on the rows/cols. Reproduces the user's
    build_matrices logic, just keyed on `lex2relation[key]` instead of POS."""
    # Restrict to lex units that have a relation assignment
    valid_indices = []
    valid_relations = []
    for idx, lex_unit in idx2lexunit.items():
        key = (lex_unit[0], lex_unit[1])
        if key in lex2relation:
            valid_indices.append(idx)
            valid_relations.append(lex2relation[key])

    relation_freqs = Counter(valid_relations)
    total = sum(relation_freqs.values())
    relation_probs = {r: c / total for r, c in relation_freqs.items()}

    valid_set = set(valid_indices)

    print(f"  computing cosine similarity over {feature_matrix.shape[0]} units...")
    sim = cosine_similarity(feature_matrix)
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)

    rows = []
    for vid, idx in enumerate(valid_indices):
        current_rel = valid_relations[vid]
        sims_row = sim[idx]
        above = np.where(sims_row >= sim_threshold)[0]
        # Keep only neighbours that are also in our valid set, sorted by sim desc
        neighbours = sorted(
            (i for i in above if i != idx and i in valid_set),
            key=lambda x: sims_row[x],
            reverse=True,
        )[:max_neighbours]
        if not neighbours:
            continue

        raw_weights = {}
        total_weight = 0.0
        for n_idx in neighbours:
            n_key = (idx2lexunit[n_idx][0], idx2lexunit[n_idx][1])
            n_rel = lex2relation[n_key]
            s = float(sims_row[n_idx])
            raw_weights[n_rel] = raw_weights.get(n_rel, 0.0) + s
            total_weight += s

        if total_weight <= 0:
            continue
        norm = {
            r: (w / total_weight) / relation_probs[r]
            for r, w in raw_weights.items()
            if relation_probs.get(r, 0) > 0
        }
        total_norm = sum(norm.values())
        if total_norm <= 0:
            continue
        for r, v in norm.items():
            rows.append({
                "Original": current_rel,
                "Neighbour": r,
                "Weight": v / total_norm,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(), relation_probs

    cats = sorted(set(df["Original"]) | set(df["Neighbour"]))
    mat = (
        df.groupby(["Original", "Neighbour"])["Weight"].sum()
        .unstack(fill_value=0.0)
        .reindex(index=cats, columns=cats, fill_value=0.0)
    )
    row_sums = mat.sum(axis=1).replace(0, 1)
    return mat.div(row_sums, axis=0), relation_probs


def rank_pairs(stability, relation_probs):
    """Build a pairwise affinity table sorted by mutual off-diagonal mass."""
    cats = stability.index.tolist()
    rows = []
    for i, a in enumerate(cats):
        for j, b in enumerate(cats):
            if i >= j:
                continue
            s_ab = float(stability.at[a, b]) if a in stability.index and b in stability.columns else 0.0
            s_ba = float(stability.at[b, a]) if b in stability.index and a in stability.columns else 0.0
            rows.append({
                "rel1": a, "rel2": b,
                "S_rel1_to_rel2": s_ab,
                "S_rel2_to_rel1": s_ba,
                "mutual_affinity": s_ab + s_ba,
                "rel1_self_coherence": float(stability.at[a, a]),
                "rel2_self_coherence": float(stability.at[b, b]),
                "rel1_freq": relation_probs.get(a, 0),
                "rel2_freq": relation_probs.get(b, 0),
            })
    return pd.DataFrame(rows).sort_values("mutual_affinity", ascending=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--treebank", required=True)
    p.add_argument("--patterns-file", required=True)
    p.add_argument("--train-conllu", required=True)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--min-dominance", type=float, default=0.5,
                   help="Lex units whose dominant relation accounts for less than "
                        "this fraction of their tokens are dropped (default 0.5).")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading corpus from {args.treebank}")
    corpus = load_corpus(args.treebank, args.patterns_file)
    feature_matrix = np.asarray(corpus.feature_matrix, dtype=np.float64)
    idx2lexunit = corpus._idx2lexunit
    print(f"  lexical units: {feature_matrix.shape[0]}, "
          f"features: {feature_matrix.shape[1]}")

    print(f"Computing dominant relation per lex unit "
          f"(min_dominance={args.min_dominance})")
    lex2relation, stats = get_dominant_relation_per_lexunit(
        args.train_conllu, idx2lexunit, args.min_dominance
    )
    print(f"  matching field: {stats['matching']}")
    print(f"  units assigned: {stats['matched_units']}")
    print(f"  dropped (low dominance):  {stats['dropped_low_dominance']}")
    rel_dist = Counter(lex2relation.values())
    print("  relation distribution (top 10):")
    for r, c in rel_dist.most_common(10):
        print(f"    {r:15s} {c}")

    print("\nBuilding relation-stability matrix")
    stability, relation_probs = build_stability_matrix(
        feature_matrix, idx2lexunit, lex2relation
    )
    if stability.empty:
        print("Empty stability matrix — check that lex units matched the conllu.")
        sys.exit(1)

    matrix_csv = os.path.join(args.output_dir, "arc_stability_matrix.csv")
    stability.round(4).to_csv(matrix_csv)
    print(f"Wrote {matrix_csv}")

    pairs_df = rank_pairs(stability, relation_probs)
    pairs_csv = os.path.join(args.output_dir, "arc_relation_affinity.csv")
    pairs_df.to_csv(pairs_csv, index=False)
    print(f"Wrote {pairs_csv}")

    # human-readable report
    diag = pd.Series(
        [stability.at[c, c] for c in stability.index],
        index=stability.index,
    ).sort_values(ascending=False)
    weights = pd.Series(relation_probs).reindex(diag.index).fillna(0)
    was = float((diag * weights).sum() / max(weights.sum(), 1e-12))

    report_path = os.path.join(args.output_dir, "arc_stability_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Arc stability report  (relations as categories)\n")
        f.write(f"Treebank: {args.treebank}\n")
        f.write(f"Units assigned a relation: {stats['matched_units']}\n")
        f.write(f"min_dominance threshold:   {args.min_dominance}\n")
        f.write(f"Weighted Average Stability (WAS): {was:.4f}\n\n")
        f.write("=== Self-coherence (diagonal), sorted desc ===\n")
        f.write(diag.round(4).to_string())
        f.write("\n\n")
        f.write("=== Top 30 mutual affinities (collapse candidates) ===\n")
        f.write("Rule of thumb: high mutual_affinity => already cluster together\n\n")
        f.write(pairs_df.head(30).round(4).to_string(index=False))
        f.write("\n\n")
        f.write("=== Bottom 15 mutual affinities (definitely distinct) ===\n")
        f.write(pairs_df.tail(15).round(4).to_string(index=False))
    print(f"Wrote {report_path}")

    print(f"\nTop 15 collapse candidates by mutual relation affinity:")
    print(pairs_df.head(15).round(4).to_string(index=False))
    print(f"\nWAS (weighted average self-coherence): {was:.4f}")


if __name__ == "__main__":
    main()
