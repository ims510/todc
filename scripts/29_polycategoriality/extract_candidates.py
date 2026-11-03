"""Extract candidate collapses from an arc-stability matrix.

A candidate collapse is a pair of dep relations (a, b) where:

  - mutual affinity S[a,b] + S[b,a] is high  (they cluster together in
    context space)
  - the target relation's diagonal is high enough that it's a stable
    landing place (we don't want to collapse INTO an unstable relation)

Direction is chosen by asymmetric pull: if a is pulled toward b more than
the reverse (S[a,b] > S[b,a] by a margin), the rule is a -> b. When both
directions pull equally, we pick the direction with the more
self-coherent target.

The output rules file uses FULL deprels (with subtypes) since the matrix
is keyed by full deprels too. So a candidate might be `obl:arg=obl:mod`
(within-base subtype collapse) or `cop=case` (cross-base collapse).

Usage:
    python extract_candidates.py path/to/stability.csv \\
        --output-dir outputs/ \\
        --min-mutual-affinity 0.3 \\
        --min-target-coherence 0.5
"""
import argparse
import os

import pandas as pd


def extract_candidates(stability, min_mutual_affinity=0.3,
                       min_target_coherence=0.5, asymmetry_ratio=1.2):
    """Identify candidate collapses.

    Args:
        stability: square DataFrame, rows = sources, cols = targets,
                   diagonal = self-coherence
        min_mutual_affinity: keep pairs whose S[a,b]+S[b,a] >= this
        min_target_coherence: target's diagonal must be at least this
        asymmetry_ratio: if S[a,b]/S[b,a] >= this, the pull is asymmetric
                         and direction is a -> b
    """
    cats = list(stability.index)
    candidates = []

    for i, a in enumerate(cats):
        for j, b in enumerate(cats):
            if i >= j:
                continue
            s_ab = float(stability.at[a, b])
            s_ba = float(stability.at[b, a])
            self_a = float(stability.at[a, a])
            self_b = float(stability.at[b, b])

            mutual = s_ab + s_ba
            if mutual < min_mutual_affinity:
                continue

            # Determine direction
            if s_ab >= s_ba * asymmetry_ratio:
                src, tgt = a, b
                direction = "asymmetric (a pulled toward b)"
            elif s_ba >= s_ab * asymmetry_ratio:
                src, tgt = b, a
                direction = "asymmetric (b pulled toward a)"
            else:
                # Symmetric: pick the less-coherent as source
                if self_a <= self_b:
                    src, tgt = a, b
                else:
                    src, tgt = b, a
                direction = "symmetric (chose by self-coherence)"

            target_coherence = float(stability.at[tgt, tgt])
            if target_coherence < min_target_coherence:
                continue

            # Categorise the candidate for interpretability
            src_coh = float(stability.at[src, src])
            tgt_coh = target_coherence
            if mutual >= 1.0 and src_coh < 0.3 and tgt_coh < 0.3:
                category = "mutual_unstable"   # neither clusters with itself, but they cluster with each other
            elif tgt_coh >= 0.5:
                category = "absorption"        # src absorbed into a stable target
            elif tgt_coh >= 0.3:
                category = "weak_absorption"
            else:
                category = "diffuse"

            candidates.append({
                "src": src,
                "tgt": tgt,
                "mutual_affinity": mutual,
                "S_src_to_tgt": float(stability.at[src, tgt]),
                "S_tgt_to_src": float(stability.at[tgt, src]),
                "src_self_coherence": src_coh,
                "tgt_self_coherence": tgt_coh,
                "direction": direction,
                "category": category,
            })

    df = pd.DataFrame(candidates)
    if df.empty:
        return df
    return df.sort_values("mutual_affinity", ascending=False).reset_index(drop=True)


def detect_chains(candidates):
    """Return set of labels appearing as both src and tgt across the rule set."""
    sources = set(candidates["src"])
    targets = set(candidates["tgt"])
    return sorted(sources & targets)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stability_csv", help="Path to stability matrix CSV")
    p.add_argument("--output-dir", default=".")
    p.add_argument("--min-mutual-affinity", type=float, default=0.3,
                   help="Minimum S[a,b]+S[b,a] for a pair to be a candidate")
    p.add_argument("--min-target-coherence", type=float, default=0.0,
                   help="Don't collapse into a target with diagonal below this. "
                        "Default 0 (disabled) — relax this if you want to allow "
                        "mutual-unstable pairs like aux:caus<->aux:pass, where "
                        "both have diagonal=0 but cluster heavily with each other "
                        "and the merged bucket would be coherent.")
    p.add_argument("--asymmetry-ratio", type=float, default=1.2,
                   help="Pull ratio above which direction is treated as asymmetric")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    stability = pd.read_csv(args.stability_csv, index_col=0)
    print(f"Loaded stability matrix: {stability.shape}")

    candidates = extract_candidates(
        stability,
        min_mutual_affinity=args.min_mutual_affinity,
        min_target_coherence=args.min_target_coherence,
        asymmetry_ratio=args.asymmetry_ratio,
    )

    if candidates.empty:
        print(f"No candidates with mutual_affinity >= {args.min_mutual_affinity} "
              f"and target self-coherence >= {args.min_target_coherence}")
        return

    print(f"\nFound {len(candidates)} candidate(s):\n")
    print(candidates.round(4).to_string(index=False))

    chains = detect_chains(candidates)
    if chains:
        print(f"\nWARNING — chain nodes (label is both src and tgt): {chains}")
        print("These rule sets need chain resolution before applying them, since")
        print("conllu collapse does one substitution per token in one pass.")

    rules_path = os.path.join(args.output_dir, "candidate_rules.txt")
    with open(rules_path, "w") as f:
        f.write(f"# Candidate rules from {args.stability_csv}\n")
        f.write(f"# Filters: min_mutual_affinity={args.min_mutual_affinity}, "
                f"min_target_coherence={args.min_target_coherence}\n")
        f.write(f"# Rules use FULL deprels (with subtypes); collapse code must "
                f"match exactly.\n\n")
        for _, row in candidates.iterrows():
            f.write(f"{row['src']}={row['tgt']}  "
                    f"# mutual={row['mutual_affinity']:.3f}  "
                    f"src_coh={row['src_self_coherence']:.3f}  "
                    f"tgt_coh={row['tgt_self_coherence']:.3f}  "
                    f"{row['direction']}\n")
        if chains:
            f.write(f"\n# CHAIN NODES (need resolution): {chains}\n")
    print(f"\nWrote {rules_path}")

    report_path = os.path.join(args.output_dir, "candidate_report.txt")
    with open(report_path, "w") as f:
        f.write("Candidate collapses from arc-stability matrix\n")
        f.write(f"Source: {args.stability_csv}\n")
        f.write(f"Filters: min_mutual_affinity={args.min_mutual_affinity}, "
                f"min_target_coherence={args.min_target_coherence}\n\n")
        f.write(candidates.round(4).to_string(index=False))
        f.write("\n\n")
        if chains:
            f.write(f"\nChain nodes: {chains}\n")
            f.write("These appear as both source and target across rules.\n")
            f.write("Before applying, decide whether to: (a) keep only the\n")
            f.write("highest-affinity rule involving each chain node, or\n")
            f.write("(b) transitively close the chain.\n")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()