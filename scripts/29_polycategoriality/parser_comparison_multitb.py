"""
multi_treebank_replication.py
=============================
Replicate the parser_comparison experiment across multiple UD treebanks
and aggregate the results.

For each treebank:
  1. Compute train-only stability scores (your existing pipeline).
  2. Run Stanza's pretrained parser on test with gold tokenization.
  3. Fit logistic regression: P(pos_err) ~ purity + log_freq.
  4. Test rival alignment using the FULL distribution (not just top-1
     rival), so high-entropy lex units are scored correctly.

Outputs:
  - One CSV per treebank with the joined token-level data.
  - summary.csv with one row per treebank.
  - forest_plot.png showing purity coefficient with 95% CI per language.
"""

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

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Point STABILITY_MODULE_DIR at the directory containing
# stability_intruder_matrix_claude.py, and TOD_DIR at your tod package.
STABILITY_MODULE_DIR = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality"
TOD_DIR              = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/tod"

OUTPUT_DIR = "./multi_treebank_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

USE_SUD = False
RIVAL_PROB_THRESHOLD = 0.05  # categories with normalized prob >= this count as rivals

# Add or remove treebanks here. Each spec needs:
#   - path:        directory containing the conllu files
#   - train_file:  filename of the training conllu
#   - test_file:   filename of the test conllu
#   - stanza_lang, stanza_pkg: passed to stanza.Pipeline
#
# Recommended starter set (download from https://universaldependencies.org/):
TREEBANKS = [
    # {"name": "fr_gsd",
    #  "path": "/Users/madalina/Documents/PHD/code/data/test/UD_French-GSD",
    #  "train_file": "fr_gsd-ud-train.conllu",
    #  "test_file":  "fr_gsd-ud-test.conllu",
    #  "stanza_lang": "fr", "stanza_pkg": "gsd"},

    # {"name": "en_ewt",  "path": "/Users/madalina/Documents/PHD/code/data/ud-treebanks-v2.17/UD_English-EWT",
    #  "train_file": "en_ewt-ud-train.conllu",
    #  "test_file":  "en_ewt-ud-test.conllu",
    #  "stanza_lang": "en", "stanza_pkg": "ewt"},

    # {"name": "ru_syntagrus", "path": "/Users/madalina/Documents/PHD/code/data/ud-treebanks-v2.17/UD_Russian-SynTagRus",
    #  "train_file": "ru_syntagrus-ud-train.conllu",
    #  "test_file":  "ru_syntagrus-ud-test.conllu",
    #  "stanza_lang": "ru", "stanza_pkg": "syntagrus"},

    {"name": "cs_pdt", "path": "/Users/madalina/Documents/PHD/code/data/ud-treebanks-v2.17/UD_Czech-PDTC",
     "train_file": "cs_pdtc-ud-train.conllu",
     "test_file":  "cs_pdtc-ud-test.conllu",
     "stanza_lang": "cs", "stanza_pkg": "pdt"},

    # {"name": "fi_tdt",  "path": "/Users/madalina/Documents/PHD/code/data/ud-treebanks-v2.17/UD_Finnish-TDT",
    #  "train_file": "fi_tdt-ud-train.conllu",
    #  "test_file":  "fi_tdt-ud-test.conllu",
    #  "stanza_lang": "fi", "stanza_pkg": "tdt"},

    # {"name": "ko_kaist", "path": "/Users/madalina/Documents/PHD/code/data/ud-treebanks-v2.17/UD_Korean-Kaist",
    #  "train_file": "ko_kaist-ud-train.conllu",
    #  "test_file":  "ko_kaist-ud-test.conllu",
    #  "stanza_lang": "ko", "stanza_pkg": "kaist"},

    # {"name": "ar_padt", "path": "/Users/madalina/Documents/PHD/code/data/ud-treebanks-v2.17/UD_Arabic-PADT",
    #  "train_file": "ar_padt-ud-train.conllu",
    #  "test_file":  "ar_padt-ud-test.conllu",
    #  "stanza_lang": "ar", "stanza_pkg": "padt"},

    {"name": "zh_gsd",  "path": "/Users/madalina/Documents/PHD/code/data/ud-treebanks-v2.17/UD_Chinese-GSD",
     "train_file": "zh_gsd-ud-train.conllu",
     "test_file":  "zh_gsd-ud-test.conllu",
     "stanza_lang": "zh-hant", "stanza_pkg": "GSD"},

    # {"name": "ja_gsd",  "path": "/Users/madalina/Documents/PHD/code/data/ud-treebanks-v2.17/UD_Japanese-GSD",
    #  "train_file": "ja_gsd-ud-train.conllu",
    #  "test_file":  "ja_gsd-ud-test.conllu",
    #  "stanza_lang": "ja", "stanza_pkg": "gsd"},

    # {"name": "tr_imst", "path": "/Users/madalina/Documents/PHD/code/data/ud-treebanks-v2.17/UD_Turkish-IMST",
    #  "train_file": "tr_imst-ud-train.conllu",
    #  "test_file":  "tr_imst-ud-test.conllu",
    #  "stanza_lang": "tr", "stanza_pkg": "imst"},

    # {"name": "hi_hdtb", "path": "/Users/madalina/Documents/PHD/code/data/ud-treebanks-v2.17/UD_Hindi-HDTB",
    #  "train_file": "hi_hdtb-ud-train.conllu",
    #  "test_file":  "hi_hdtb-ud-test.conllu",
    #  "stanza_lang": "hi", "stanza_pkg": "hdtb"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_distribution(dist_str):
    """Parse 'ADJ: 0.45, NOUN: 0.30' into {'ADJ': 0.45, 'NOUN': 0.30}."""
    if not dist_str or pd.isna(dist_str):
        return {}
    out = {}
    for piece in str(dist_str).split(","):
        piece = piece.strip()
        if ":" not in piece:
            continue
        cat, prob = piece.rsplit(":", 1)
        try:
            out[cat.strip()] = float(prob.strip())
        except ValueError:
            continue
    return out


def compute_train_scores(spec):
    """Run your existing scoring pipeline on the train portion only."""
    if STABILITY_MODULE_DIR not in sys.path:
        sys.path.insert(0, STABILITY_MODULE_DIR)
    if TOD_DIR not in sys.path:
        sys.path.insert(0, TOD_DIR)
    from stability_intruder_matrix_claude import (
        GREW_PATTERN, PATTERNS_TEXT_FILE,
        EXCLUDED_FEATURE_PATTERNS, INCLUDED_FEATURE_PATTERNS,
        build_similarity_matrix, compute_scores,
    )
    import tod.corpus

    train_path = os.path.join(spec["path"], spec["train_file"])
    tmp = tempfile.mkdtemp(prefix=f"{spec['name']}_train_")
    shutil.copy(train_path, os.path.join(tmp, spec["train_file"]))

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

    token_freqs = Counter()
    with open(train_path, encoding="utf-8") as f:
        for sent in conllu.parse_incr(f):
            for tok in sent:
                if isinstance(tok["id"], int):
                    token_freqs[(tok["lemma"], tok["upos"])] += 1
    return scores, token_freqs


def run_parser(spec):
    """Parse test set with Stanza using gold tokenization."""
    stanza.download(spec["stanza_lang"], package=spec["stanza_pkg"], verbose=False)
    nlp = stanza.Pipeline(
        lang=spec["stanza_lang"], package=spec["stanza_pkg"],
        processors='tokenize,pos,lemma,depparse',
        tokenize_pretokenized=True, use_gpu=False, verbose=False,
    )
    test_path = os.path.join(spec["path"], spec["test_file"])
    with open(test_path, encoding="utf-8") as f:
        gold = list(conllu.parse_incr(f))
    pretok    = [[t["form"] for t in s if isinstance(t["id"], int)] for s in gold]
    input_str = "\n".join(" ".join(toks) for toks in pretok)
    doc = nlp(input_str)

    parsed = []
    for gs, ps in zip(gold, doc.sentences):
        gtoks = [t for t in gs if isinstance(t["id"], int)]
        if len(gtoks) != len(ps.words):
            continue
        rows = []
        for g, p in zip(gtoks, ps.words):
            ext = (g.get("misc") or {}).get("ExtPos")
            rows.append({
                "lemma":     g["lemma"],
                "form":      g["form"],
                "gold_upos": ext or g["upos"],
                "pred_upos": p.upos,
            })
        parsed.append(rows)
    return parsed


def build_dataframe(parsed, scores, token_freqs):
    rows = []
    for sent in parsed:
        for tok in sent:
            key  = (tok["lemma"], tok["gold_upos"])
            sc   = scores.get(key)
            freq = token_freqs.get(key, 0)
            rows.append({
                **tok,
                "purity":       sc["purity"]         if sc else None,
                "entropy":      sc["entropy"]        if sc else None,
                "rival":        sc["rival_category"] if sc else None,
                "distribution": sc["distribution"]   if sc else None,
                "in_train":     freq > 0,
                "train_freq":   freq,
            })
    df = pd.DataFrame(rows)
    df["pos_err"]  = (df["pred_upos"] != df["gold_upos"]).astype(int)
    df["log_freq"] = np.log1p(df["train_freq"])
    return df


def fit_regression(df):
    sub = df.dropna(subset=["purity"]).copy()
    if len(sub) < 50 or sub["pos_err"].sum() < 5:
        return None
    try:
        m = smf.logit("pos_err ~ purity + log_freq", data=sub).fit(disp=0)
        ci = m.conf_int().loc["purity"]
        return {
            "n":           len(sub),
            "n_errors":    int(sub["pos_err"].sum()),
            "coef":        float(m.params["purity"]),
            "se":          float(m.bse["purity"]),
            "ci_low":      float(ci.iloc[0]),
            "ci_high":     float(ci.iloc[1]),
            "p":           float(m.pvalues["purity"]),
            "log_freq_coef": float(m.params["log_freq"]),
        }
    except Exception as e:
        print(f"  regression failed: {e}")
        return None


def rival_alignment(df, threshold=RIVAL_PROB_THRESHOLD):
    """
    For each POS error, check whether pred_upos is among the lex unit's
    *meaningful rivals* — every category with normalized mass >= threshold,
    excluding the gold category itself. This handles high-entropy cases
    where there are several legitimate rivals, not just one.
    """
    sub = df.dropna(subset=["distribution", "purity", "entropy"]).copy()
    if len(sub) == 0:
        return None

    def in_rivals(row):
        d = parse_distribution(row["distribution"])
        rivals = {c for c, p in d.items()
                  if p >= threshold and c != row["gold_upos"]}
        return row["pred_upos"] in rivals

    sub["pred_in_rivals"] = sub.apply(in_rivals, axis=1)

    # Median splits → cells
    pur_med = sub["purity"].median()
    ent_med = sub["entropy"].median()
    sub["pur_lvl"] = np.where(sub["purity"]  <  pur_med, "low_pur",  "high_pur")
    sub["ent_lvl"] = np.where(sub["entropy"] >= ent_med, "high_ent", "low_ent")
    sub["cell"]    = sub["pur_lvl"] + " × " + sub["ent_lvl"]

    errs = sub[sub["pos_err"] == 1]
    if len(errs) == 0:
        return {"n_errors": 0, "overall_hit_rate": np.nan}

    out = {"n_errors": int(len(errs)),
           "overall_hit_rate": float(errs["pred_in_rivals"].mean())}
    for cell in ["low_pur × low_ent", "low_pur × high_ent",
                 "high_pur × low_ent", "high_pur × high_ent"]:
        ce = errs[errs["cell"] == cell]
        out[f"hit_{cell}"] = (float(ce["pred_in_rivals"].mean())
                              if len(ce) else np.nan)
        out[f"n_{cell}"]   = int(len(ce))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    summary = []
    for spec in TREEBANKS:
        print(f"\n{'='*60}\n{spec['name']}\n{'='*60}")
        try:
            scores, freqs = compute_train_scores(spec)
            print(f"  {len(scores)} lex units scored")
            parsed = run_parser(spec)
            n_tok = sum(len(s) for s in parsed)
            print(f"  {n_tok} test tokens parsed")

            df = build_dataframe(parsed, scores, freqs)
            df.to_csv(os.path.join(OUTPUT_DIR, f"{spec['name']}_tokens.csv"),
                      index=False)

            reg = fit_regression(df)
            ra  = rival_alignment(df)

            row = {"treebank": spec["name"], "n_test_tokens": n_tok}
            if reg: row.update(reg)
            if ra:  row.update(ra)
            summary.append(row)

            if reg:
                print(f"  purity coef = {reg['coef']:+.3f} "
                      f"[{reg['ci_low']:+.2f}, {reg['ci_high']:+.2f}]  "
                      f"p={reg['p']:.1e}")
            if ra:
                print(f"  rival hit rate (errors): "
                      f"{ra['overall_hit_rate']:.2%} of {ra['n_errors']}")
        except Exception as e:
            import traceback; traceback.print_exc()
            summary.append({"treebank": spec["name"], "error": str(e)})

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))

    # ------------------- Forest plot -------------------
    valid = summary_df.dropna(subset=["coef"]).sort_values("coef")
    if len(valid) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(valid))))
    y = np.arange(len(valid))
    xerr = np.vstack([valid["coef"] - valid["ci_low"],
                      valid["ci_high"] - valid["coef"]])
    colors = ["green" if hi < 0 else "red" if lo > 0 else "gray"
              for lo, hi in zip(valid["ci_low"], valid["ci_high"])]
    for yi, (xi, lo, hi, c) in enumerate(zip(valid["coef"],
                                             valid["ci_low"],
                                             valid["ci_high"],
                                             colors)):
        ax.errorbar(xi, yi, xerr=[[xi - lo], [hi - xi]],
                    fmt='o', color=c, ecolor=c, capsize=4)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(valid["treebank"])
    ax.set_xlabel("Purity coefficient (95% CI), controlling for log_freq")
    ax.set_title("Effect of stability on POS-tagging error across treebanks")
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "forest_plot.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved forest plot: {out}")


if __name__ == "__main__":
    main()