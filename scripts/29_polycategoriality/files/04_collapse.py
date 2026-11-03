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
import argparse
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


def parse_rules(rules_str: str) -> Dict[str, str]:
    rules: Dict[str, str] = {}
    for item in rules_str.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid collapse rule: {item}. Expected format base=target")
        base, target = item.split("=", 1)
        rules[base.strip()] = target.strip()
    return rules


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collapse relation labels in a CoNLL-U file using a mapping of base relations."
    )
    parser.add_argument("input_path", help="Input CoNLL-U file")
    parser.add_argument("output_path", help="Output CoNLL-U file")
    parser.add_argument(
        "--rules",
        default=None,
        help="Optional collapse rules, e.g. 'obl=nmod,csubj=nsubj'."
    )
    args = parser.parse_args()

    rules = parse_rules(args.rules) if args.rules else None
    stats = collapse_conllu(args.input_path, args.output_path, rules)
    print(f"Wrote {args.output_path}")
    print(f"Total tokens: {stats.pop('total')}")
    print(f"Changed:      {stats.pop('changed')}")
    for rule, n in stats.items():
        print(f"  {rule}: {n}")
