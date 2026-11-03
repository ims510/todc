"""Stage 4: rewrite a CoNLL-U file with collapsed relation labels.

`collapse_conllu` takes a path to a CoNLL-U file and a mapping
{base_relation: collapsed_relation} and writes a new file where every
matching relation in column 8 is replaced. Subtypes after ':' are
preserved (e.g. nsubj:pass becomes csubj:pass under {'nsubj': 'csubj'}).

After running stages 1-3 you may want to extend DEFAULT_COLLAPSE_RULES.
The default below collapses obl into nmod, which on UD English-EWT had
majority_acc ≈ 0.984 and cond_entropy ≈ 0.09 — the classic case where
head_upos is enough to recover the original label.

Usage:
    python 04_collapse.py <input.conllu> <output.conllu>
"""
import sys
from typing import Dict, Optional

DEFAULT_COLLAPSE_RULES: Dict[str, str] = {
    "obl": "nmod",
    # Add more after running stage 3 if you find them, e.g.
    # "csubj": "nsubj",
}


def collapse_conllu(
    input_path: str,
    output_path: str,
    rules: Optional[Dict[str, str]] = None,
) -> Dict[str, int]:
    """Apply the rules to column 8 (DEPREL). Returns counts."""
    if rules is None:
        rules = DEFAULT_COLLAPSE_RULES
    n_changed = 0
    n_total = 0
    by_rule = {k: 0 for k in rules}

    with open(input_path, encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if line.startswith("#") or not line.strip():
                fout.write(line)
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 10:
                fout.write(line)
                continue
            if "-" in fields[0] or "." in fields[0]:
                fout.write(line)
                continue
            n_total += 1
            deprel = fields[7]
            base, _, suffix = deprel.partition(":")
            if base in rules:
                new_base = rules[base]
                fields[7] = f"{new_base}:{suffix}" if suffix else new_base
                n_changed += 1
                by_rule[base] += 1
            fout.write("\t".join(fields) + "\n")

    return {"total": n_total, "changed": n_changed, **by_rule}


def _read_rules_file(path):
    """Read 'src=tgt' lines from a rules file. '# comments' stripped, blanks skipped."""
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


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument(
        "--rules",
        default=None,
        help="Comma-separated collapse rules, e.g. 'obl=nmod,csubj=nsubj'.",
    )
    p.add_argument(
        "--rules-file",
        default=None,
        help="Path to a rules file (one 'src=tgt' per line, # comments OK). "
        "Overrides --rules if both are given. If neither is given, "
        "DEFAULT_COLLAPSE_RULES is used.",
    )
    args = p.parse_args()

    if args.rules_file:
        rules = _read_rules_file(args.rules_file)
    elif args.rules:
        rules = {}
        for part in args.rules.split(","):
            part = part.strip()
            if not part:
                continue
            a, b = part.split("=")
            rules[a.strip()] = b.strip()
    else:
        rules = None

    stats = collapse_conllu(args.input, args.output, rules)
    print(f"Wrote {args.output}")
    print(f"Total tokens: {stats.pop('total')}")
    print(f"Changed:      {stats.pop('changed')}")
    for rule, n in stats.items():
        print(f"  {rule}: {n}")
