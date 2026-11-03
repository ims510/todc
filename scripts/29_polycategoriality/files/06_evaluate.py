"""Stage 6: evaluate parser predictions and (optionally) deterministically
uncollapse them using head POS, then re-evaluate against the original gold.

UAS = % of (non-punct) tokens with correct head.
LAS = % with correct head AND correct DEPREL (base label, ignoring ':' subtype).

Two evaluation modes:
  (a) Direct: compare predictions to a gold conllu directly.
  (b) Uncollapse: predictions were made in the COLLAPSED label scheme;
      apply a deterministic rule {(collapsed_label, head_upos) -> orig_label}
      to recover the original label, then evaluate against the ORIGINAL gold.

The uncollapse rules are derived from the original training file: for every
(collapsed_label, head_upos) cell we record the majority original label.
This is the "deterministic given POS" claim from stage 3 cashed out.

Usage:
    python 06_evaluate.py compare <gold.conllu> <pred.conllu>
    python 06_evaluate.py uncollapse <orig_train.conllu> <orig_test.conllu> \\
                                     <pred_collapsed.conllu> <out_uncollapsed.conllu> \\
                                     [--rules obl=nmod,...]
"""
import argparse
import sys
from collections import Counter, defaultdict
from typing import Dict, Tuple

from utils import parse_conllu, base_relation, get_head_upos


def parse_rules(s: str) -> Dict[str, str]:
    rules = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split("=")
        rules[a.strip()] = b.strip()
    return rules


def derive_uncollapse_rules(
    train_path: str, collapse_rules: Dict[str, str]
) -> Dict[Tuple[str, str], str]:
    """For each (collapsed_label, head_upos), find the majority ORIGINAL label.

    Walks the original (uncollapsed) training file and counts which original
    base relation each (collapsed_label, head_upos) cell is dominated by.
    """
    affected_collapsed = set(collapse_rules.values())
    counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for sent in parse_conllu(train_path):
        for tok in sent:
            orig = base_relation(tok["deprel"])
            collapsed = collapse_rules.get(orig, orig)
            if collapsed not in affected_collapsed:
                continue
            h_upos = get_head_upos(sent, tok)
            counts[(collapsed, h_upos)][orig] += 1
    return {k: ctr.most_common(1)[0][0] for k, ctr in counts.items()}


def apply_uncollapse(
    pred_path: str,
    out_path: str,
    rules: Dict[Tuple[str, str], str],
    fallback_rules: Dict[str, str],
):
    """Rewrite predicted file: for each token, look up (label, head_upos)
    in `rules` and replace label if matched. The head's UPOS comes from the
    PREDICTED head (we look it up in the same predicted sentence). For tokens
    whose label is in fallback_rules.values() but has no matching rule cell,
    we leave the prediction unchanged."""
    sentences = list(parse_conllu(pred_path))
    affected = set(fallback_rules.values())
    n_changed = 0
    n_total = 0

    # We rewrite line by line using original file order so all comments
    # are preserved. We also need a per-sentence map id->upos.
    sent_iter = iter(sentences)
    cur_sent = None
    cur_map = None
    with open(pred_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.startswith("#"):
                fout.write(line)
                continue
            if not line.strip():
                fout.write(line)
                cur_sent = None
                cur_map = None
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 10 or "-" in fields[0] or "." in fields[0]:
                fout.write(line)
                continue
            if cur_sent is None:
                cur_sent = next(sent_iter)
                cur_map = {t["id"]: t["upos"] for t in cur_sent}
            n_total += 1
            base, _, suffix = fields[7].partition(":")
            if base in affected:
                try:
                    head_id = int(fields[6])
                except ValueError:
                    head_id = 0
                head_upos = "ROOT" if head_id <= 0 else cur_map.get(head_id, "ROOT")
                rule_target = rules.get((base, head_upos))
                if rule_target is not None and rule_target != base:
                    fields[7] = f"{rule_target}:{suffix}" if suffix else rule_target
                    n_changed += 1
            fout.write("\t".join(fields) + "\n")
    return n_total, n_changed


def score(gold_path: str, pred_path: str):
    """UAS / LAS, excluding punct (standard UD eval). Compares base label
    only (everything before ':')."""
    gold_sents = list(parse_conllu(gold_path))
    pred_sents = list(parse_conllu(pred_path))
    if len(gold_sents) != len(pred_sents):
        raise ValueError(
            f"Sentence count mismatch: gold {len(gold_sents)} vs pred {len(pred_sents)}"
        )
    n = uas = las = 0
    label_correct = Counter()
    label_total = Counter()
    confusion = defaultdict(Counter)  # gold_label -> Counter[pred_label]
    for gs, ps in zip(gold_sents, pred_sents):
        if len(gs) != len(ps):
            raise ValueError("Token count mismatch in a sentence")
        for gt, pt in zip(gs, ps):
            g_rel = base_relation(gt["deprel"])
            p_rel = base_relation(pt["deprel"])
            if g_rel == "punct":
                continue
            n += 1
            label_total[g_rel] += 1
            confusion[g_rel][p_rel] += 1
            if pt["head"] == gt["head"]:
                uas += 1
                if p_rel == g_rel:
                    las += 1
                    label_correct[g_rel] += 1
    return {
        "n": n,
        "UAS": uas / max(1, n),
        "LAS": las / max(1, n),
        "per_label": {
            r: (label_correct[r], label_total[r], label_correct[r] / label_total[r])
            for r in sorted(label_total)
        },
        "confusion": confusion,
    }


def print_scores(name: str, s: dict, focus_labels=None):
    print(f"\n=== {name} ===")
    print(f"n_tokens (excl. punct): {s['n']}")
    print(f"UAS: {s['UAS']:.4f}")
    print(f"LAS: {s['LAS']:.4f}")
    if focus_labels:
        print("\nFocus-label LAS-recall:")
        for r in focus_labels:
            if r in s["per_label"]:
                c, t, a = s["per_label"][r]
                print(f"  {r:12s} {c}/{t} = {a:.4f}")


def write_score_report(path: str, header: str, s: dict, extra: str = ""):
    with open(path, "w") as f:
        f.write(header + "\n")
        f.write(f"n_tokens (excl. punct): {s['n']}\n")
        f.write(f"UAS: {s['UAS']:.4f}\n")
        f.write(f"LAS: {s['LAS']:.4f}\n")
        f.write("\nPer-label LAS-recall:\n")
        for r, (c, t, a) in sorted(s["per_label"].items(), key=lambda x: -x[1][1]):
            f.write(f"  {r:14s} {c}/{t} = {a:.4f}\n")
        if extra:
            f.write("\n" + extra + "\n")


def cmd_compare(args):
    s = score(args.gold, args.pred)
    print_scores(f"{args.pred} vs {args.gold}", s, focus_labels=["nmod", "obl", "nsubj", "csubj"])
    if args.out:
        write_score_report(args.out, f"{args.pred} vs {args.gold}", s)
        print(f"\nWrote {args.out}")


def cmd_uncollapse(args):
    rules_collapse = parse_rules(args.rules)
    print(f"Collapse rules: {rules_collapse}")
    uncollapse_rules = derive_uncollapse_rules(args.orig_train, rules_collapse)
    print(f"Derived {len(uncollapse_rules)} uncollapse cells")
    rule_lines = []
    for k, v in sorted(uncollapse_rules.items()):
        line = f"  ({k[0]:>6s}, {k[1]:>6s}) -> {v}"
        rule_lines.append(line)
        print(line)
    n_total, n_changed = apply_uncollapse(
        args.pred_collapsed, args.out_uncollapsed, uncollapse_rules, rules_collapse
    )
    print(f"\nApplied uncollapse: {n_changed}/{n_total} tokens changed")
    s = score(args.orig_test, args.out_uncollapsed)
    print_scores(
        f"uncollapsed predictions vs {args.orig_test}",
        s,
        focus_labels=["nmod", "obl", "nsubj", "csubj"],
    )
    if args.out:
        extra = (
            f"Collapse rules: {rules_collapse}\n"
            f"Uncollapse cells:\n" + "\n".join(rule_lines) + "\n"
            f"Tokens changed by uncollapse: {n_changed}/{n_total}"
        )
        write_score_report(args.out, f"uncollapsed predictions vs {args.orig_test}", s, extra)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("compare")
    pc.add_argument("gold")
    pc.add_argument("pred")
    pc.add_argument("--out", help="Optional report file")
    pc.set_defaults(func=cmd_compare)

    pu = sub.add_parser("uncollapse")
    pu.add_argument("orig_train", help="Original (uncollapsed) train.conllu — used to derive rules")
    pu.add_argument("orig_test", help="Original (uncollapsed) test.conllu — used as gold")
    pu.add_argument("pred_collapsed", help="Predictions in collapsed label scheme")
    pu.add_argument("out_uncollapsed", help="Output: predictions after uncollapse")
    pu.add_argument("--rules", default="obl=nmod",
                    help="Collapse rules used during training, e.g. 'obl=nmod,csubj=nsubj'")
    pu.add_argument("--out", help="Optional report file")
    pu.set_defaults(func=cmd_uncollapse)

    args = p.parse_args()
    args.func(args)
