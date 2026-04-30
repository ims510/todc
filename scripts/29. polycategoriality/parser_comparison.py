"""
parser_comparison.py
====================
Test whether low-purity/high-entropy lexical units are disproportionately
misparsed by an off-the-shelf parser on UD_French-GSD.

Hypothesis
----------
After controlling for token frequency, P(parser error) is higher for
tokens whose (lemma, upos) lex unit had low purity (or high entropy)
in the train-derived stability analysis.

Pipeline
--------
  Phase 1 — compute stability scores on TRAIN only (avoids leakage).
  Phase 2 — run Stanza's pretrained fr_gsd model on TEST with gold
            tokenization, so predictions align 1:1 with gold tokens.
  Phase 3 — per test token, look up purity/entropy for (lemma, gold_upos).
  Phase 4 — analysis: OOV breakdown, Spearman, logistic regressions,
            decile plot, per-POS table.

Dependencies
------------
  pip install stanza conllu statsmodels pandas numpy scipy matplotlib

Run
---
  python parser_comparison.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections import Counter

import conllu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stanza
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# CONFIG — adjust paths to your setup
# ---------------------------------------------------------------------------
TREEBANK_DIR = "/Users/madalina/Documents/PHD/code/data/test/UD_French-GSD"
TRAIN_FILE   = os.path.join(TREEBANK_DIR, "fr_gsd-ud-train.conllu")
TEST_FILE    = os.path.join(TREEBANK_DIR, "fr_gsd-ud-test.conllu")

# Where your stability_intruder_matrix_claude.py and `tod` package live
STABILITY_MODULE_DIR = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality"
TOD_DIR              = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/tod"

OUTPUT_DIR = "./parser_comparison_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# IMPORTANT: For this experiment, use UD (not SUD) so the stability
# scores and the parser predictions are in the same schema.
USE_SUD = False


# ---------------------------------------------------------------------------
# Phase 1 — stability scores from TRAIN ONLY
# ---------------------------------------------------------------------------
def compute_train_scores():
    """
    Reuses your existing pipeline (`tod.corpus.Corpus`, `compute_scores`)
    on a temp directory containing only the training CoNLL-U file.

    Returns
    -------
    scores      : dict (lemma, upos) -> {purity, entropy, rival_category, ...}
    token_freqs : Counter (lemma, upos) -> int (token count in train)
    """
    sys.path.insert(0, STABILITY_MODULE_DIR)
    sys.path.insert(0, TOD_DIR)
    from stability_intruder_matrix_claude import (
        GREW_PATTERN, PATTERNS_TEXT_FILE,
        EXCLUDED_FEATURE_PATTERNS, INCLUDED_FEATURE_PATTERNS,
        build_similarity_matrix, compute_scores,
    )
    import tod.corpus

    # Build a temp dir with ONLY the train file
    tmp = tempfile.mkdtemp(prefix="udfg_train_")
    shutil.copy(TRAIN_FILE, os.path.join(tmp, os.path.basename(TRAIN_FILE)))

    corpus = tod.corpus.Corpus(
        treebank_path=tmp,
        grew_pattern=GREW_PATTERN,
        patterns_text_file=PATTERNS_TEXT_FILE,
        use_sud=USE_SUD,
        matrix_type="coverage",
        excluded_feature_patterns=EXCLUDED_FEATURE_PATTERNS,
        included_feature_patterns=INCLUDED_FEATURE_PATTERNS,
    )
    sim    = build_similarity_matrix(corpus)
    scores = compute_scores(corpus, sim)

    # Token frequencies in train (for the regression control)
    token_freqs = Counter()
    with open(TRAIN_FILE, encoding="utf-8") as f:
        for sent in conllu.parse_incr(f):
            for tok in sent:
                if isinstance(tok["id"], int):  # skip MWT meta-tokens
                    token_freqs[(tok["lemma"], tok["upos"])] += 1

    return scores, token_freqs


# ---------------------------------------------------------------------------
# Phase 2 — run Stanza on TEST, with gold tokenization
# ---------------------------------------------------------------------------
def run_parser_on_test():
    """
    Parse TEST with Stanza's pretrained French-GSD model using gold
    tokenization (so predictions align 1:1 with gold tokens). MWT is
    DISABLED because we feed Stanza already-expanded sub-tokens.

    Returns
    -------
    list of sentence rows (each a list of per-token dicts)
    """
    stanza.download('fr', package='gsd', verbose=False)
    nlp = stanza.Pipeline(
        lang='fr',
        package='gsd',
        processors='tokenize,pos,lemma,depparse',
        tokenize_pretokenized=True,
        use_gpu=False,
        verbose=False,
    )

    with open(TEST_FILE, encoding="utf-8") as f:
        gold_sents = list(conllu.parse_incr(f))

    # Pretokenized input: one sentence per line, tokens space-separated
    pretok = [
        [t["form"] for t in sent if isinstance(t["id"], int)]
        for sent in gold_sents
    ]
    input_str = "\n".join(" ".join(toks) for toks in pretok)
    doc = nlp(input_str)

    if len(doc.sentences) != len(gold_sents):
        print(f"[warning] sentence count mismatch: gold={len(gold_sents)} "
              f"stanza={len(doc.sentences)}")

    parsed = []
    for sent_idx, (gold_sent, pred_sent) in enumerate(zip(gold_sents, doc.sentences)):
        gold_tokens = [t for t in gold_sent if isinstance(t["id"], int)]
        if len(gold_tokens) != len(pred_sent.words):
            print(f"[skip sent {sent_idx}] token count mismatch "
                  f"{len(gold_tokens)} vs {len(pred_sent.words)}")
            continue
        sent_rows = []
        for gtok, pword in zip(gold_tokens, pred_sent.words):
            sent_rows.append({
                "sent_idx":    sent_idx,
                "id":          int(gtok["id"]),
                "form":        gtok["form"],
                "lemma":       gtok["lemma"],
                "gold_upos":   gtok["upos"],
                "gold_head":   gtok["head"],
                "gold_deprel": (gtok["deprel"] or "").split(":")[0],
                "pred_upos":   pword.upos,
                "pred_head":   pword.head,
                "pred_deprel": (pword.deprel or "").split(":")[0],
            })
        parsed.append(sent_rows)
    return parsed


# ---------------------------------------------------------------------------
# Phase 3 — join: each test token gets train-derived stability + freq
# ---------------------------------------------------------------------------
def build_token_dataframe(parsed_test, scores, token_freqs):
    rows = []
    for sent in parsed_test:
        for tok in sent:
            key   = (tok["lemma"],  tok.get("misc", {}).get("ExtPos") or tok["gold_upos"])
            score = scores.get(key)
            freq  = token_freqs.get(key, 0)
            rows.append({
                **tok,
                "purity":    score["purity"]         if score else None,
                "entropy":   score["entropy"]        if score else None,
                "rival":     score["rival_category"] if score else None,
                "in_train":  freq > 0,
                "has_score": score is not None,
                "train_freq": freq,
            })
    df = pd.DataFrame(rows)

    # Error flags
    df["pos_err"] = (df["pred_upos"] != df["gold_upos"]).astype(int)
    df["uas_err"] = (df["pred_head"] != df["gold_head"]).astype(int)
    df["las_err"] = ((df["pred_head"]   != df["gold_head"]) |
                     (df["pred_deprel"] != df["gold_deprel"])).astype(int)
    df["log_freq"] = np.log1p(df["train_freq"])
    return df


# ---------------------------------------------------------------------------
# Phase 4 — analysis
# ---------------------------------------------------------------------------
def run_analysis(df):
    n_total = len(df)
    n_in_train = df["in_train"].sum()
    n_scored   = df["has_score"].sum()

    print(f"\nTest tokens total           : {n_total}")
    print(f"  seen in train (vocab)     : {n_in_train} "
          f"({100*n_in_train/n_total:.1f}%)")
    print(f"  with stability score      : {n_scored} "
          f"({100*n_scored/n_total:.1f}%)")

    # ------------------------------------------------------------------
    # Sanity check: OOV vs in-vocab error rates
    # ------------------------------------------------------------------
    print("\n=== OOV vs in-vocab error rates ===")
    print(df.groupby("in_train")[["pos_err", "uas_err", "las_err"]]
            .mean().to_string())

    # ------------------------------------------------------------------
    # Logistic regression: P(error) ~ purity + log_freq
    # ------------------------------------------------------------------
    print("\n=== Logistic regression: error ~ purity + log_freq ===")
    print("Hypothesis confirmed if `purity` coef is negative and significant\n"
          "*after* controlling for log_freq.")
    for outcome in ["pos_err", "uas_err", "las_err"]:
        try:
            m = smf.logit(f"{outcome} ~ purity + log_freq",
                          data=sub).fit(disp=0)
            print(f"\n--- {outcome} ---")
            print(m.summary().tables[1])
        except Exception as e:
            print(f"[{outcome}] fit failed: {e}")

    # ------------------------------------------------------------------
    # Decile plot
    # ------------------------------------------------------------------
    sub["purity_decile"] = pd.qcut(sub["purity"], q=10,
                                    labels=False, duplicates="drop")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, outcome in zip(axes, ["pos_err", "uas_err", "las_err"]):
        binned = sub.groupby("purity_decile")[outcome].mean()
        ax.bar(binned.index, binned.values, color="steelblue")
        ax.set_title(f"{outcome} by purity decile (0=lowest)")
        ax.set_xlabel("Purity decile")
        ax.set_ylabel("Mean error rate")
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "decile_plot.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved decile plot: {out_path}")

    # ------------------------------------------------------------------
    # Per-POS breakdown — useful for the paper
    # ------------------------------------------------------------------
    print("\n=== Per-gold-UPOS POS-error rate, sorted by mean purity ===")
    grp = (sub.groupby("gold_upos")
              .agg(n=("pos_err", "size"),
                   pos_err=("pos_err", "mean"),
                   mean_purity=("purity", "mean"))
              .sort_values("mean_purity"))
    print(grp.to_string())

    return sub


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Phase 1 — train-only stability scores …")
    scores, token_freqs = compute_train_scores()
    print(f"  {len(scores)} lex units scored, "
          f"{len(token_freqs)} (lemma, upos) types in train")

    print("\nPhase 2 — parsing test set with Stanza fr_gsd …")
    parsed_test = run_parser_on_test()
    print(f"  Parsed {sum(len(s) for s in parsed_test)} test tokens")

    print("\nPhase 3 — joining …")
    df = build_token_dataframe(parsed_test, scores, token_freqs)
    csv_path = os.path.join(OUTPUT_DIR, "tokens_joined.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved per-token data: {csv_path}")

    print("\nPhase 4 — analysis …")
    sub = run_analysis(df)