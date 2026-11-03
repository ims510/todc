"""Shared CoNLL-U parsing helpers."""
from typing import Iterator, List, Dict


def parse_conllu(path: str) -> Iterator[List[Dict]]:
    """Yield sentences as lists of token dicts.

    Skips comment lines, multi-word token ranges (id contains '-'),
    and empty nodes (id contains '.'). These are not used for
    dependency analysis.
    """
    with open(path, encoding="utf-8") as f:
        sent: List[Dict] = []
        for raw in f:
            line = raw.rstrip("\n")
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
            tok_id = fields[0]
            if "-" in tok_id or "." in tok_id:
                continue
            try:
                head = int(fields[6]) if fields[6] != "_" else -1
                tid = int(tok_id)
            except ValueError:
                continue
            sent.append(
                {
                    "id": tid,
                    "form": fields[1],
                    "lemma": fields[2],
                    "upos": fields[3],
                    "xpos": fields[4],
                    "feats": fields[5],
                    "head": head,
                    "deprel": fields[7],
                    "deps": fields[8],
                    "misc": fields[9],
                }
            )
        if sent:
            yield sent


def base_relation(deprel: str) -> str:
    """Return the part before ':'. e.g. 'nsubj:pass' -> 'nsubj'."""
    return deprel.split(":", 1)[0]


def get_head_upos(sent: List[Dict], tok: Dict) -> str:
    """UPOS of token's head, or 'ROOT' if attached to virtual root."""
    h = tok["head"]
    if h <= 0:
        return "ROOT"
    # Token IDs are 1-based and contiguous after we filter MWT/empty nodes.
    # Build a quick id->index lookup if needed; for typical UD this is fine.
    for t in sent:
        if t["id"] == h:
            return t["upos"]
    return "ROOT"


def iter_arcs(path: str):
    """Yield (head_upos, dep_upos, base_relation, full_deprel, sent, tok)."""
    for sent in parse_conllu(path):
        for tok in sent:
            rel = base_relation(tok["deprel"])
            h_upos = get_head_upos(sent, tok)
            yield h_upos, tok["upos"], rel, tok["deprel"], sent, tok
