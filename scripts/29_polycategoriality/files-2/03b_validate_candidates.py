"""Stage 3b: Validate stage-3 candidates with the category-purity test.

For each candidate rule (r1 -> r2):
  1. Apply the collapse as a COLUMN MERGE on the feature matrix.
     Every feature column matching rel_shallow=r1[:subtype] is added into
     the corresponding rel_shallow=r2[:subtype] column. Subtypes are
     preserved: obl:agent -> nmod:agent (a new column is created if the
     target doesn't exist yet). This is equivalent to relabelling the
     treebank and rebuilding the feature matrix, but ~100x cheaper.
  2. Recompute the cosine-similarity stability matrix from the merged matrix.
  3. Score the change against the original stability matrix:
       ΔWAS                  - change in weighted average stability (scalar)
       worst_category_loss   - largest single drop on the diagonal
       worst_category        - which POS category took that hit
       max_offdiag_increase  - largest increase in any cross-category cell

To calibrate both metrics against a null distribution, we sample
random (src, tgt) collapses from the corpus's relations and compute their
scores. A candidate is declared SAFE if BOTH conditions hold:
  - Its worst_category_loss >= null p10
    ("not in worst 10% for category damage")
  - Its max_offdiag_increase <= null p90
    ("not in worst 10% for cross-category mixing")

Usage:
    python 03b_validate_candidates.py \\
        --treebank /path/to/UD_French-GSD \\
        --patterns-file /path/to/patterns_all_nodes.txt \\
        --candidates outputs/03_suggested_rules.txt \\
        --output-dir outputs \\
        --null-samples 50 \\
        --null-percentile 10

Requires the user's `tod` library on the import path. Set TOD_PATH env var.
"""
import argparse
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Make `tod` importable. Override with TOD_PATH env var if your install
# lives somewhere else.
# _TOD_DEFAULT = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/tod"
# _TOD_PATH = os.environ.get("TOD_PATH", _TOD_DEFAULT)
# if os.path.isdir(_TOD_PATH):
#     sys.path.insert(1, _TOD_PATH)

import sys
sys.path.insert(1, "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/tod")
import tod.corpus # noqa: E402


# These mirror the user's existing stability_intruder_matrix.py config.
SIM_THRESHOLD = 0.7
MAX_NEIGHBOURS = 20

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


# --------------------------------------------------------------------------
# Corpus loading
# --------------------------------------------------------------------------

def load_corpus(treebank_path, patterns_file,
                grew_pattern="pattern{X[upos<>PUNCT]}"):
    return tod.corpus.Corpus(
        treebank_path=treebank_path,
        grew_pattern=grew_pattern,
        patterns_text_file=patterns_file,
        use_sud=False,
        matrix_type="coverage",
        excluded_feature_patterns=EXCLUDED_FEATURE_PATTERNS,
        included_feature_patterns=INCLUDED_FEATURE_PATTERNS,
    )


def global_category_freqs(idx2lexunit):
    all_tags = [lu[1] for lu in idx2lexunit.values()]
    total = len(all_tags)
    return {cat: count / total for cat, count in Counter(all_tags).items()}


# --------------------------------------------------------------------------
# Column-merge collapse
# --------------------------------------------------------------------------

# Parse a feature name like 'node:X:child:rel_shallow=obl:agent' into:
#   prefix   = 'node:X:child:rel_shallow='   (everything up to and including 'rel_shallow=')
#   base     = 'obl'                          (the base relation)
#   subtype  = ':agent'                       (optional, may be empty)
#   tail     = ''                             (rarely non-empty; whatever follows)
#
# This lets a base-level rule (obl -> nmod) be applied to every variant:
#   rel_shallow=obl        -> rel_shallow=nmod
#   rel_shallow=obl:agent  -> rel_shallow=nmod:agent   (new column if needed)
#   rel_shallow=obl:arg    -> rel_shallow=nmod:arg
#   rel_shallow=obl:mod    -> rel_shallow=nmod:mod
_REL_FEAT_RE = re.compile(
    r"^(.*?rel_shallow=)"
    r"([A-Za-z][A-Za-z_]*)"
    r"(:[A-Za-z][A-Za-z_]*)?"
    r"(.*)$"
)


def parse_rel_feature(name):
    """Return (prefix, base, subtype_with_colon_or_empty, tail) or None."""
    m = _REL_FEAT_RE.match(name)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3) or "", m.group(4)


def apply_column_merge(feature_matrix, idx2feature, collapse_rules):
    """Apply rules src -> tgt by merging feature columns.

    Subtypes are preserved: rel_shallow=obl:agent under rule obl->nmod
    becomes rel_shallow=nmod:agent. Target columns that don't exist yet
    are created (zero-initialised and then filled by the merge).

    Returns:
        new_matrix          - copy of feature_matrix with merges applied
        new_idx2feature     - dict including any newly added columns
        n_merged            - count of source columns that were merged
        n_new_columns       - count of target columns that had to be created
    """
    feat2idx = {n: i for i, n in idx2feature.items()}
    new_idx2feature = dict(idx2feature)

    # Plan: list (src_idx, tgt_name) pairs
    operations = []
    for idx, name in idx2feature.items():
        parsed = parse_rel_feature(name)
        if parsed is None:
            continue
        prefix, base, subtype, tail = parsed
        if base not in collapse_rules:
            continue
        tgt_base = collapse_rules[base]
        tgt_name = f"{prefix}{tgt_base}{subtype}{tail}"
        operations.append((idx, tgt_name))

    # Add columns for any targets that don't exist
    new_target_names = sorted({t for _, t in operations if t not in feat2idx})
    n_new = len(new_target_names)
    if n_new > 0:
        M = np.hstack([
            feature_matrix,
            np.zeros((feature_matrix.shape[0], n_new), dtype=feature_matrix.dtype),
        ])
        next_idx = feature_matrix.shape[1]
        for tgt_name in new_target_names:
            feat2idx[tgt_name] = next_idx
            new_idx2feature[next_idx] = tgt_name
            next_idx += 1
    else:
        M = feature_matrix.copy()

    # Apply merges
    for src_idx, tgt_name in operations:
        tgt_idx = feat2idx[tgt_name]
        M[:, tgt_idx] += M[:, src_idx]
        M[:, src_idx] = 0.0

    return M, new_idx2feature, len(operations), n_new


def list_base_relations(idx2feature):
    """All distinct base relations seen in rel_shallow features."""
    bases = set()
    for name in idx2feature.values():
        parsed = parse_rel_feature(name)
        if parsed is not None:
            bases.add(parsed[1])
    return sorted(bases)


# --------------------------------------------------------------------------
# Stability matrix (reimplemented from user's build_matrices)
# --------------------------------------------------------------------------

def build_stability_matrix(feature_matrix, idx2lexunit, category_freqs,
                           sim_threshold=SIM_THRESHOLD,
                           max_neighbours=MAX_NEIGHBOURS):
    """Reproduces the stability matrix from stability_intruder_matrix.py.

    Difference: this version takes the feature matrix as a parameter
    instead of reading it from a Corpus object, so we can test collapses
    cheaply without rebuilding the corpus.
    """
    sim = cosine_similarity(feature_matrix)
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)

    rows = []
    for idx, lex_unit in idx2lexunit.items():
        current_cat = lex_unit[1]
        sims_row = sim[idx]
        above = np.where(sims_row >= sim_threshold)[0]
        neighbours = sorted(
            (i for i in above if i != idx),
            key=lambda x: sims_row[x],
            reverse=True,
        )[:max_neighbours]
        if not neighbours:
            continue

        raw_weights = {}
        total_weight = 0.0
        for n_idx in neighbours:
            n_cat = idx2lexunit[n_idx][1]
            s = float(sims_row[n_idx])
            raw_weights[n_cat] = raw_weights.get(n_cat, 0.0) + s
            total_weight += s

        if total_weight <= 0:
            continue

        # Frequency-normalise then re-normalise to sum to 1
        norm = {
            cat: (w / total_weight) / category_freqs[cat]
            for cat, w in raw_weights.items()
            if category_freqs.get(cat, 0) > 0
        }
        total_norm = sum(norm.values())
        if total_norm <= 0:
            continue
        p_norm = {cat: v / total_norm for cat, v in norm.items()}

        for cat, prob in p_norm.items():
            rows.append({"Original": current_cat, "Neighbour": cat, "Weight": prob})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    cats = sorted(set(df["Original"]) | set(df["Neighbour"]))
    mat = (
        df.groupby(["Original", "Neighbour"])["Weight"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(index=cats, columns=cats, fill_value=0.0)
    )
    row_sums = mat.sum(axis=1).replace(0, 1)
    return mat.div(row_sums, axis=0)


def weighted_average_stability(stability, category_freqs):
    cats = stability.index
    diag = pd.Series(np.diag(stability), index=cats)
    weights = pd.Series(category_freqs).reindex(cats).fillna(0)
    total = weights.sum()
    return float((diag * weights).sum() / total) if total > 0 else 0.0


# --------------------------------------------------------------------------
# Candidate scoring
# --------------------------------------------------------------------------

def parse_candidate_file(path):
    """Read 'src=tgt' lines from a rules file. Strips '# comments' and blanks."""
    rules = []
    with open(path) as f:
        for line in f:
            if "#" in line:
                line = line[: line.index("#")]
            line = line.strip()
            if not line or "=" not in line:
                continue
            src, tgt = line.split("=", 1)
            rules.append((src.strip(), tgt.strip()))
    return rules


def score_collapse(rule_dict, feature_matrix, idx2feature,
                   idx2lexunit, category_freqs,
                   stability_orig, was_orig):
    """Score a collapse rule dict {src: tgt} (one or many at once).

    Returns a dict of diagnostics, or None if the rule didn't touch any
    relation column in the feature matrix.
    """
    M, _, n_applied, n_new = apply_column_merge(feature_matrix, idx2feature, rule_dict)
    if n_applied == 0:
        return None

    stability_new = build_stability_matrix(M, idx2lexunit, category_freqs)
    was_new = weighted_average_stability(stability_new, category_freqs)

    cats = sorted(set(stability_orig.index) | set(stability_new.index))
    s_orig = stability_orig.reindex(index=cats, columns=cats, fill_value=0)
    s_new = stability_new.reindex(index=cats, columns=cats, fill_value=0)
    diff = s_new - s_orig

    delta_diag = pd.Series(np.diag(diff), index=cats)
    off_diag = diff.copy()
    np.fill_diagonal(off_diag.values, 0)

    return {
        "delta_was": was_new - was_orig,
        "was_new": was_new,
        "worst_category": str(delta_diag.idxmin()),
        "worst_category_loss": float(delta_diag.min()),
        "best_category": str(delta_diag.idxmax()),
        "best_category_gain": float(delta_diag.max()),
        "max_offdiag_increase": float(off_diag.values.max()),
        "n_columns_merged": n_applied,
        "n_columns_created": n_new,
        "delta_diag": delta_diag.to_dict(),
    }


# --------------------------------------------------------------------------
# Null distribution
# --------------------------------------------------------------------------

def build_null_distribution(feature_matrix, idx2feature, idx2lexunit, category_freqs,
                            stability_orig, was_orig, relation_pool,
                            n_samples=50, seed=42, exclude_pairs=None):
    """Score a sample of RANDOM collapses to build a null distribution of ΔWAS.

    For each sample, picks a random (src, tgt) pair from relation_pool (both
    distinct, src != tgt) and applies that collapse. Returns a list of dicts
    {src, tgt, delta_was, worst_category_loss, max_offdiag_increase}.

    exclude_pairs: iterable of (src, tgt) tuples to skip (typically the real
                   candidates, so the null is a clean reference).
    """
    import random
    rng = random.Random(seed)
    exclude = set(exclude_pairs or [])
    pool = list(relation_pool)
    null = []
    attempts = 0
    while len(null) < n_samples and attempts < n_samples * 10:
        attempts += 1
        src = rng.choice(pool)
        tgt = rng.choice([r for r in pool if r != src])
        if (src, tgt) in exclude or (tgt, src) in exclude:
            continue
        sc = score_collapse(
            {src: tgt}, feature_matrix, idx2feature,
            idx2lexunit, category_freqs, stability_orig, was_orig,
        )
        if sc is None:
            continue
        null.append({
            "src": src, "tgt": tgt,
            "delta_was": sc["delta_was"],
            "worst_category_loss": sc["worst_category_loss"],
            "max_offdiag_increase": sc["max_offdiag_increase"],
        })
    return null


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--treebank", required=True,
                   help="Path to UD treebank directory")
    p.add_argument("--patterns-file", required=True,
                   help="Path to grew patterns_all_nodes.txt")
    p.add_argument("--candidates", default="outputs/03_suggested_rules.txt",
                   help="Stage-3 candidate rules file")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--was-threshold", type=float, default=0.005,
                   help="|ΔWAS| <= this is safe")
    p.add_argument("--diag-threshold", type=float, default=0.02,
                   help="worst_category_loss >= -this is safe")
    p.add_argument("--offdiag-threshold", type=float, default=0.02,
                   help="max_offdiag_increase <= this is safe")
    p.add_argument("--also-test-combined", action="store_true",
                   help="After scoring individually, also test the combined "
                        "effect of all SAFE rules together")
    p.add_argument("--null-samples", type=int, default=50,
                   help="Number of random collapses to sample for the null "
                        "distribution. 0 disables null calibration.")
    p.add_argument("--null-percentile", type=float, default=10.0,
                   help="A candidate is calibrated-safe if its |ΔWAS| is at or "
                        "below this percentile of the null distribution. "
                        "Default 10 means 'less disruptive than 90%% of random "
                        "collapses'.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading corpus: {args.treebank}")
    corpus = load_corpus(args.treebank, args.patterns_file)
    feature_matrix = np.asarray(corpus.feature_matrix, dtype=np.float64)
    idx2lexunit = corpus._idx2lexunit
    idx2feature = corpus._idx2feature
    print(f"  lexical units: {feature_matrix.shape[0]}")
    print(f"  features:      {feature_matrix.shape[1]}")

    base_relations = list_base_relations(idx2feature)
    print(f"  distinct base relations in rel_shallow features: {len(base_relations)}")
    if not base_relations:
        print("WARNING: no 'rel_shallow=' feature columns found in the corpus.")
        print("Inspect a few feature names and adjust _REL_FEAT_RE if needed.")
        sys.exit(1)

    category_freqs = global_category_freqs(idx2lexunit)

    print("Building baseline stability matrix...")
    stability_orig = build_stability_matrix(
        feature_matrix, idx2lexunit, category_freqs
    )
    was_orig = weighted_average_stability(stability_orig, category_freqs)
    print(f"  baseline WAS: {was_orig:.4f}")

    # ----- score each candidate -----
    candidates = parse_candidate_file(args.candidates)
    print(f"\nScoring {len(candidates)} candidates from {args.candidates}\n")

    results = []
    for i, (src, tgt) in enumerate(candidates, 1):
        sc = score_collapse(
            {src: tgt},
            feature_matrix, idx2feature, idx2lexunit, category_freqs,
            stability_orig, was_orig,
        )
        if sc is None:
            print(f"  [{i}/{len(candidates)}] {src}->{tgt}: SKIP "
                  "(no matching rel_shallow columns)")
            continue
        print(f"  [{i}/{len(candidates)}] {src:>10s} -> {tgt:<10s}  "
              f"ΔWAS={sc['delta_was']:+.5f}  "
              f"worst={sc['worst_category']}:{sc['worst_category_loss']:+.4f}  "
              f"max_off={sc['max_offdiag_increase']:+.4f}  "
              f"n_merges={sc['n_columns_merged']}+{sc['n_columns_created']}new")
        results.append({"src": src, "tgt": tgt, **sc})

    # ----- null distribution -----
    null_pct = None
    null_records = []
    if args.null_samples > 0 and base_relations:
        print(f"\nBuilding null distribution from {args.null_samples} random collapses...")
        excl = [(r["src"], r["tgt"]) for r in results]
        null_records = build_null_distribution(
            feature_matrix, idx2feature, idx2lexunit, category_freqs,
            stability_orig, was_orig,
            relation_pool=base_relations,
            n_samples=args.null_samples,
            seed=args.seed,
            exclude_pairs=excl,
        )
        if null_records:
            abs_dwa = [abs(n["delta_was"]) for n in null_records]
            worst_loss = [n["worst_category_loss"] for n in null_records]
            null_pct = {
                "abs_delta_was_pX": float(np.percentile(abs_dwa, args.null_percentile)),
                "abs_delta_was_p50": float(np.percentile(abs_dwa, 50)),
                "abs_delta_was_p90": float(np.percentile(abs_dwa, 90)),
                "worst_loss_p10": float(np.percentile(worst_loss, 10)),
                "worst_loss_p50": float(np.percentile(worst_loss, 50)),
                "max_offdiag_p90": float(np.percentile(
                    [n["max_offdiag_increase"] for n in null_records], 90
                )),
            }
            print(f"  null |ΔWAS| at p{int(args.null_percentile):d}={null_pct['abs_delta_was_pX']:.5f}, "
                  f"p50={null_pct['abs_delta_was_p50']:.5f}, "
                  f"p90={null_pct['abs_delta_was_p90']:.5f}")

    # ----- decide which candidates are safe -----
    if null_pct is not None:
        # Calibrated: candidate passes if both metrics are within safe ranges
        diag_thresh = null_pct["worst_loss_p10"]
        offdiag_thresh = null_pct["max_offdiag_p90"]
    else:
        diag_thresh = -args.diag_threshold
        offdiag_thresh = args.offdiag_threshold

    for r in results:
        r["is_safe"] = (
            r["worst_category_loss"] >= diag_thresh
            and r["max_offdiag_increase"] <= offdiag_thresh
        )

    safe_rules = [r for r in results if r["is_safe"]]

    # ----- write outputs -----
    csv_path = os.path.join(args.output_dir, "03b_validation_results.csv")
    pd.DataFrame(
        [{k: v for k, v in r.items() if k != "delta_diag"} for r in results]
    ).to_csv(csv_path, index=False)

    if null_records:
        null_csv = os.path.join(args.output_dir, "03b_null_distribution.csv")
        pd.DataFrame(null_records).to_csv(null_csv, index=False)

    rules_path = os.path.join(args.output_dir, "03b_validated_rules.txt")
    with open(rules_path, "w") as f:
        f.write(
            "# Validated collapse rules from stage 3b (purity-preservation test).\n"
            f"# baseline WAS = {was_orig:.4f}\n"
            f"# worst_category_loss threshold (p10): {diag_thresh:.4f} "
            f"({'null-calibrated' if null_pct else 'fixed'})\n"
            f"# max_offdiag_increase threshold (p90): {offdiag_thresh:.4f} "
            f"({'null-calibrated' if null_pct else 'fixed'})\n"
            f"# candidates: {len(results)}    validated: {len(safe_rules)}\n\n"
        )
        for r in sorted(safe_rules, key=lambda x: abs(x["delta_was"])):
            f.write(
                f"{r['src']}={r['tgt']}    "
                f"# ΔWAS={r['delta_was']:+.5f} "
                f"worst={r['worst_category']}:{r['worst_category_loss']:+.4f}\n"
            )

    report_path = os.path.join(args.output_dir, "03b_validation_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Baseline WAS: {was_orig:.4f}\n")
        f.write(f"Treebank:     {args.treebank}\n\n")
        if null_pct is not None:
            f.write("Null distribution (random src->tgt collapses):\n")
            f.write(f"  samples:  {len(null_records)}\n")
            f.write(f"  worst_category_loss percentiles:  "
                    f"p10={null_pct['worst_loss_p10']:.4f}  "
                    f"p50={null_pct['worst_loss_p50']:.4f}\n")
            f.write(f"  max_offdiag_increase p90={null_pct['max_offdiag_p90']:.4f}\n\n")
            f.write(f"Safety filters (null-calibrated):\n")
            f.write(f"  worst_category_loss >= null p10 = {diag_thresh:.4f}\n")
            f.write(f"  max_offdiag_increase <= null p90 = {offdiag_thresh:.4f}\n\n")
        else:
            f.write(f"worst_category_loss threshold (fixed): {diag_thresh:.4f}\n")
            f.write(f"max_offdiag_increase threshold (fixed): {offdiag_thresh:.4f}\n\n")
        f.write("All candidates ranked by |ΔWAS|:\n\n")
        for r in sorted(results, key=lambda r: abs(r["delta_was"])):
            flag = "[SAFE]" if r["is_safe"] else "[FAIL]"
            f.write(
                f"{flag} {r['src']:>10s} -> {r['tgt']:<10s}  "
                f"ΔWAS={r['delta_was']:+.5f}  "
                f"worst={r['worst_category']:>6s}={r['worst_category_loss']:+.4f}  "
                f"best={r['best_category']:>6s}={r['best_category_gain']:+.4f}  "
                f"max_off={r['max_offdiag_increase']:+.4f}  "
                f"merges={r['n_columns_merged']}\n"
            )

    # ----- optionally test the combined effect of all safe rules -----
    if args.also_test_combined and safe_rules:
        combined = {r["src"]: r["tgt"] for r in safe_rules}
        combined_score = score_collapse(
            combined,
            feature_matrix, idx2feature, idx2lexunit, category_freqs,
            stability_orig, was_orig,
        )
        with open(report_path, "a") as f:
            f.write("\n\nCombined effect of all SAFE rules applied together:\n")
            f.write(f"  rules: {combined}\n")
            f.write(f"  ΔWAS = {combined_score['delta_was']:+.5f}\n")
            f.write(f"  worst category: {combined_score['worst_category']} "
                    f"({combined_score['worst_category_loss']:+.4f})\n")
            f.write(f"  max offdiag:    {combined_score['max_offdiag_increase']:+.4f}\n")
        print(f"\nCombined ΔWAS = {combined_score['delta_was']:+.5f}  "
              f"(worst={combined_score['worst_category']}:"
              f"{combined_score['worst_category_loss']:+.4f})")

    print(f"\nWrote {csv_path}")
    print(f"Wrote {rules_path}")
    print(f"Wrote {report_path}")
    print(f"\nValidated {len(safe_rules)}/{len(results)} candidates.")
    print(f"Pass to stage 4 with:  --rules-file {rules_path}")


if __name__ == "__main__":
    main()