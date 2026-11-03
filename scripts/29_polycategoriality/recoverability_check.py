"""Recoverability check for a candidate ruleset.

Given a rules file (src=tgt per line, # comments OK) and a training conllu,
this script measures how often the deterministic uncollapse from stage 6 can
recover the original label, using head_upos + dep_upos as the cue (with a
head-only fallback).

If the overall accuracy is high (say >95%), the rule set preserves the
information needed for deterministic recovery and the parser experiment in
stages 4-6 will succeed. If the accuracy is low, chains or POS-cell overlaps
have destroyed the information and the parser experiment will see per-label
LAS drops at uncollapse time — visible HERE without training a parser.

This is the same accuracy that stage 3 already computes for pairs (a, b); we
just apply it to the FULL ruleset to spot chain damage.

Usage:
    python recoverability_check.py <rules_file> <train.conllu> [--out report.txt]
"""
import argparse
import os
from collections import Counter, defaultdict


def base_relation(deprel):
    return deprel.split(":", 1)[0]


def get_label(deprel, use_base=False):
    """Return the label to match against rules. When use_base=False
    (the default for arc-stability candidates), the full deprel including
    subtype is used so that rules like 'obl:arg=obl:mod' work as expected.
    When use_base=True, only the part before ':' is used — matches the
    previous behaviour of stage-4 collapse."""
    return base_relation(deprel) if use_base else deprel


def read_rules(path):
    rules = {}
    with open(path) as f:
        for line in f:
            if "#" in line:
                line = line[: line.index("#")]
            line = line.strip()
            if not line or "=" not in line:
                continue
            src, tgt = line.split("=", 1)
            rules[src.strip()] = tgt.strip()
    return rules


def walk_conllu(path):
    with open(path, encoding="utf-8") as f:
        sent = []
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if not line.strip():
                if sent:
                    yield sent
                    sent = []
                continue
            fields = line.split("\t")
            if len(fields) != 10:
                continue
            if "-" in fields[0] or "." in fields[0]:
                continue
            try:
                sent.append({
                    "id": int(fields[0]),
                    "upos": fields[3],
                    "head": int(fields[6]) if fields[6] != "_" else 0,
                    "deprel": fields[7],
                })
            except ValueError:
                continue
        if sent:
            yield sent


def head_upos(sent, tok):
    h = tok["head"]
    if h <= 0:
        return "ROOT"
    for t in sent:
        if t["id"] == h:
            return t["upos"]
    return "ROOT"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rules_file")
    p.add_argument("train_conllu")
    p.add_argument("--out", default=None)
    p.add_argument("--use-base", action="store_true",
                   help="Match rules and labels using only the base relation "
                        "(part before ':'). By default, full deprels including "
                        "subtypes are used, which is what you want for "
                        "arc-stability candidates like 'obl:arg=obl:mod'.")
    args = p.parse_args()

    rules = read_rules(args.rules_file)
    if not rules:
        print("No rules in file. Nothing to do.")
        return
    print(f"Loaded {len(rules)} rules from {args.rules_file}")
    print(f"  rules: {rules}")
    print(f"  matching: {'base relation only' if args.use_base else 'full deprel including subtypes'}\n")

    affected_collapsed = set(rules.values())

    # First pass: build the majority-class maps from the training data.
    cell_counts = defaultdict(Counter)
    fallback_counts = defaultdict(Counter)
    instances = []
    for sent in walk_conllu(args.train_conllu):
        for tok in sent:
            original = get_label(tok["deprel"], use_base=args.use_base)
            collapsed = rules.get(original, original)
            if collapsed not in affected_collapsed:
                continue
            h = head_upos(sent, tok)
            d = tok["upos"]
            cell_counts[(collapsed, h, d)][original] += 1
            fallback_counts[(collapsed, h)][original] += 1
            instances.append((original, collapsed, h, d))

    full_rules = {k: ctr.most_common(1)[0][0] for k, ctr in cell_counts.items()}
    fallback_rules = {k: ctr.most_common(1)[0][0] for k, ctr in fallback_counts.items()}

    # Second pass: score recoverability of each instance
    total = len(instances)
    correct = 0
    per_original = defaultdict(lambda: [0, 0])
    confusion = defaultdict(Counter)
    for original, collapsed, h, d in instances:
        recovered = (
            full_rules.get((collapsed, h, d))
            or fallback_rules.get((collapsed, h))
            or collapsed
        )
        per_original[original][1] += 1
        confusion[original][recovered] += 1
        if recovered == original:
            correct += 1
            per_original[original][0] += 1

    overall = correct / max(1, total)
    worst_label = None
    worst_acc = 1.0
    for orig, (c, t) in per_original.items():
        if t < 50:  # ignore tiny populations
            continue
        a = c / t
        if a < worst_acc:
            worst_acc = a
            worst_label = orig

    # ---- print + write report ----
    lines = []
    lines.append(f"Rules: {rules}")
    lines.append(f"Tokens evaluated: {total}")
    lines.append(f"Overall recoverability accuracy: {correct}/{total} = {overall:.4f}")
    if worst_label is not None:
        lines.append(f"Worst-label recoverability:      {worst_label} = {worst_acc:.4f}")
    lines.append("")
    lines.append("Per-original-label recoverability:")
    lines.append(f"  {'label':<12s} {'recovered':>10s} {'total':>8s}  {'acc':>6s}  most-common-confusion")
    for orig, (c, t) in sorted(per_original.items(), key=lambda x: -x[1][1]):
        acc = c / max(1, t)
        wrong = [(r, n) for r, n in confusion[orig].items() if r != orig]
        wrong.sort(key=lambda x: -x[1])
        confusion_str = ""
        if wrong:
            r, n = wrong[0]
            confusion_str = f"  -> {r} ({n} times, {n/max(1,t):.2f})"
        lines.append(f"  {orig:<12s} {c:>10d} {t:>8d}  {acc:>6.4f}{confusion_str}")
    lines.append("")
    overall_ok = overall > 0.95
    per_label_ok = worst_acc > 0.90 if worst_label else True
    if overall_ok and per_label_ok:
        lines.append("Verdict: OK")
    elif overall_ok and not per_label_ok:
        lines.append(f"Verdict: WARNING — overall is high but {worst_label} "
                     f"recoverability is only {worst_acc:.2f}.")
        lines.append("  Likely cause: chains, or two collapsed-together relations")
        lines.append("  share head_upos x dep_upos cells. Parser uncollapse will")
        lines.append("  drop LAS for the affected label.")
    else:
        lines.append("Verdict: FAIL — overall recoverability is lossy.")

    print("\n".join(lines))
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()