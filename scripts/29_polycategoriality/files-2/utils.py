"""Shared CoNLL-U parsing helpers."""
from typing import Iterator, List, Dict, Iterable, Optional

# Relations and POS that carry no useful information for the
# "is this distinction redundant given POS" analysis. Stages 1 and 3
# skip arcs matching either set.
#
#   punct / root: structurally trivial, dominate the top-TVD list with
#                 pairs that never share any (h_upos, d_upos) cell.
#   PUNCT dep:    same arcs as the punct relation viewed from the POS side.
DEFAULT_EXCLUDE_RELATIONS = {"punct", "root"}
DEFAULT_EXCLUDE_DEP_UPOS = {"PUNCT"}


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


def should_skip(
    tok: Dict,
    sent: Optional[List[Dict]] = None,
    exclude_rels: Optional[Iterable[str]] = None,
    exclude_upos: Optional[Iterable[str]] = None,
) -> bool:
    """Return True if this arc should be skipped from the redundancy analysis.

    Filters out, by default:
      - tokens whose base relation is in `exclude_rels` (default: punct, root)
      - tokens whose own UPOS is in `exclude_upos`  (default: PUNCT)
      - if `sent` is given, also tokens whose HEAD'S UPOS is in `exclude_upos`
        (so arcs attached to punctuation are dropped too)

    Pass empty sets/frozensets to disable.
    """
    if exclude_rels is None:
        exclude_rels = DEFAULT_EXCLUDE_RELATIONS
    if exclude_upos is None:
        exclude_upos = DEFAULT_EXCLUDE_DEP_UPOS
    if base_relation(tok["deprel"]) in exclude_rels:
        return True
    if tok["upos"] in exclude_upos:
        return True
    if sent is not None:
        if get_head_upos(sent, tok) in exclude_upos:
            return True
    return False