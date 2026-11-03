"""
confirm_constructions.py
========================
Ground-truth confirmation for diachronic candidates, independent of the
distributional pipeline. For a target lemma+POS and a construction matcher,
it counts — per bin — how often the lemma occurs and how often it occurs in
the construction, straight from the raw CoNLL-U. If a distributional candidate
is real, its construction rate should move; if it's a register/parse artifact,
it won't.

Matchers are LINEAR (they look at neighbouring tokens' forms/lemmas) rather
than dependency-based, because the constructions of interest are fixed
multiword sequences and linear matching sidesteps Stanza's parse errors.

Usage
-----
    bins = build_bin_groups()                       # [(label, [conllu paths])]
    df = construction_frequency(bins, "partir", "VERB", m_a_partir_de)
    print(df)        # bin, n_occ, n_hits, rate, per_1k_tokens
    plot_rate(df, "partir : a partir de")
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

METADATA = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/31. diachronic_polycateg/TRIAD/metadata.tsv"
TEXT_DIR = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/31. diachronic_polycateg/TRIAD/Textes_tagged_stanza"
BIN_BY   = "decennie"


# ---------------------------------------------------------------------------
# CoNLL-U reading
# ---------------------------------------------------------------------------

def _feats(s):
    if s in ("_", ""):
        return {}
    return dict(kv.split("=", 1) for kv in s.split("|") if "=" in kv)


def iter_sentences(path):
    """Yield each sentence as a list of token dicts."""
    sent = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                if sent:
                    yield sent
                    sent = []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 10 or ("-" in cols[0]) or ("." in cols[0]):
                continue
            sent.append({
                "id": int(cols[0]), "form": cols[1], "lemma": cols[2],
                "upos": cols[3], "feats": _feats(cols[5]),
                "head": cols[6], "deprel": cols[7],
            })
    if sent:
        yield sent


def build_bin_groups(metadata_path=METADATA, text_dir=TEXT_DIR, by=BIN_BY):
    df = pd.read_csv(metadata_path, sep="\t")
    df["path"] = df["name"].apply(lambda n: os.path.join(text_dir, f"{n}.conllu"))
    grouped = df.groupby(by)["path"].apply(list)
    return [(int(label), [p for p in paths if os.path.exists(p)])
            for label, paths in sorted(grouped.items())]


# ---------------------------------------------------------------------------
# The counter
# ---------------------------------------------------------------------------

def construction_frequency(bin_groups, lemma, upos, matcher):
    """
    For each bin, count occurrences of (lemma, upos) and how many satisfy
    `matcher(sent, i)`. Returns a DataFrame with the raw counts and two rates:
      rate          = hits / occurrences of the lemma   (is the lemma being
                      used in this construction more often?)
      per_1k_tokens = hits / total tokens * 1000         (is the construction
                      itself getting more frequent in the corpus?)
    """
    rows = []
    for label, paths in bin_groups:
        n_occ = n_hits = n_tokens = 0
        for p in paths:
            for sent in iter_sentences(p):
                n_tokens += len(sent)
                for i, tok in enumerate(sent):
                    if tok["lemma"].lower() == lemma.lower() and tok["upos"] == upos:
                        n_occ += 1
                        if matcher(sent, i):
                            n_hits += 1
        rows.append({
            "bin": label, "n_occ": n_occ, "n_hits": n_hits,
            "rate": (n_hits / n_occ) if n_occ else 0.0,
            "per_1k_tokens": (n_hits / n_tokens * 1000) if n_tokens else 0.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ready-made matchers (lower-cased, form- and lemma-tolerant)
# ---------------------------------------------------------------------------

def _lemmas_before(sent, i, n):
    return [sent[j]["lemma"].lower() for j in range(max(0, i - n), i)]

def _forms_before(sent, i, n):
    return [sent[j]["form"].lower() for j in range(max(0, i - n), i)]

def _next(sent, i):
    return sent[i + 1] if i + 1 < len(sent) else None

def _is_de(tok):
    return tok is not None and (tok["lemma"].lower() == "de"
                                or tok["form"].lower() in {"de", "du", "des", "d'"})


def m_dans_le_cadre_de(sent, i):
    """dans/en le cadre de ..."""
    before = _lemmas_before(sent, i, 3)
    return any(l in {"dans", "en"} for l in before) and _is_de(_next(sent, i))


def m_au_niveau_de(sent, i):
    """au/du niveau de ..."""
    before = _forms_before(sent, i, 3) + _lemmas_before(sent, i, 3)
    return any(l in {"au", "du", "à", "a"} for l in before) and _is_de(_next(sent, i))


def m_a_partir_de(sent, i):
    """à partir de ..."""
    before = _forms_before(sent, i, 2) + _lemmas_before(sent, i, 2)
    return any(l in {"à", "a"} for l in before) and _is_de(_next(sent, i))


def m_compte_participle(sent, i):
    """compte rendu / compte tenu (compte + past participle)."""
    nxt = _next(sent, i)
    if nxt is None:
        return False
    if nxt["form"].lower() in {"rendu", "rendus", "tenu", "tenue", "tenus"}:
        return True
    return nxt["upos"] in {"VERB", "ADJ"} and nxt["feats"].get("VerbForm") == "Part"

def m_de_noun(sent, i):
    """... de <noun>  (preceding de/du/des/d')"""
    prev = sent[i - 1] if i > 0 else None
    return _is_de(prev)

def m_en_noun(sent, i):
    """en <noun>  (preceding en)"""
    prev = sent[i - 1] if i > 0 else None
    return prev is not None and prev["lemma"].lower() == "en"

def m_a_venir(sent, i):
    """<noun> à venir  (preceding à)"""
    prev = sent[i - 1] if i > 0 else None
    return prev is not None and (prev["lemma"].lower() in {"à", "a"}
                                 or prev["form"].lower() in {"à", "a"})

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_rate(df, title, save=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["bin"], df["rate"], marker="o", label="share of lemma's uses")
    ax.plot(df["bin"], df["per_1k_tokens"], marker="s", label="per 1k tokens")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Bin"); ax.set_ylabel("Frequency")
    ax.set_title(title); ax.legend(); fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    plt.show(); plt.close()


if __name__ == "__main__":
    bins = build_bin_groups()
    checks = [
        # ("partir", "VERB", m_a_partir_de,        "partir : à partir de"),
        # ("compte", "NOUN", m_compte_participle,   "compte : compte rendu/tenu"),
        # ("niveau", "NOUN", m_au_niveau_de,        "niveau : au niveau de"),
        # ("cadre",  "NOUN", m_dans_le_cadre_de,    "cadre : dans le cadre de"),
        # ("partenariat", "NOUN", m_en_noun,     "partenariat : en partenariat"),
        # ("coopération", "NOUN", m_en_noun,     "coopération : en coopération"),
        ("partenariat", "NOUN", m_de_noun,     "partenariat : de partenariat"),
        ("coopération", "NOUN", m_de_noun,     "coopération : de coopération"),
        ("venir", "VERB", m_a_venir,     "venir : à venir"),
    ]
    for lemma, upos, matcher, title in checks:
        df = construction_frequency(bins, lemma, upos, matcher)
        print(f"\n{title}\n{df.to_string(index=False)}")
        plot_rate(df, title, save=f"construction_{lemma}.png")