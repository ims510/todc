"""
parser_comparison_multitb.py
============================
Replicate the parser_comparison experiment across all eligible UD treebanks
and aggregate the results.

Eligibility (auto-discovered from --ud-root):
  * train + test conllu files exist
  * lemmas are annotated (sampled check)
  * >= --min-tokens combined train+test tokens
  * Stanza has tokenize/POS/depparse models for the language (default
    package per processor; lemma falls back to identity when missing)
  * one treebank per Stanza language (the largest is kept)

Caching:
  Each treebank's intermediate artefacts live under CACHE_DIR/<name>/:

      spec.json    the resolved spec for this run
      scores.pkl   stability scores from the train-only pipeline
      freqs.pkl    Counter of (lemma, upos) -> train freq
      parsed.pkl   Stanza output, aligned to gold
      tokens.csv   joined per-token dataframe (the analysis input)

  The expensive steps (scores, parser) are skipped if their .pkl is
  already on disk. Use --force to recompute.

  Use --analyze-only to skip discovery + processing entirely and just
  rebuild summary.csv and forest_plot.png from the cached tokens.csvs.
  This is what you want when iterating on plots / regressions.

  Use --dry-run to print the discovery result and exit.
  Use --only to restrict to specific treebank names.

Outputs (in --output-dir):
  summary.csv      one row per treebank
  forest_plot.png  purity coef + 95% CI per treebank
"""

import argparse
import json
import os
import pickle
import shutil
import sys
import tempfile
import traceback
from collections import Counter

import conllu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ---------------------------------------------------------------------------
# CONFIG -- edit these for your environment
# ---------------------------------------------------------------------------
STABILITY_MODULE_DIR = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality"
TOD_DIR              = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/tod"

UD_ROOT    = "/Users/madalina/Documents/PHD/code/data/ud-treebanks-v2.17"
CACHE_DIR  = "./multi_treebank_cache"
OUTPUT_DIR = "./multi_treebank_output"

MIN_TOKENS           = 10_000
LEMMA_COVERAGE_MIN   = 0.5
USE_SUD              = False
RIVAL_PROB_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def count_tokens(conllu_path):
    """Fast token count: skip comments, multi-word tokens, empty nodes."""
    n = 0
    with open(conllu_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line[0] == "#":
                continue
            first = line.split("\t", 1)[0]
            if "-" in first or "." in first:
                continue
            n += 1
    return n


def has_lemmas(conllu_path, sample=2000, threshold=LEMMA_COVERAGE_MIN):
    """True if at least `threshold` of sampled tokens have a real lemma."""
    seen = with_lemma = 0
    with open(conllu_path, encoding="utf-8") as f:
        for sent in conllu.parse_incr(f):
            for tok in sent:
                if not isinstance(tok["id"], int):
                    continue
                seen += 1
                lem = tok.get("lemma")
                if lem and lem != "_":
                    with_lemma += 1
                if seen >= sample:
                    break
            if seen >= sample:
                break
    return seen > 0 and with_lemma / seen >= threshold


def load_stanza_index():
    """Return Stanza's resources.json as a dict (downloads on first call)."""
    import stanza
    from stanza.resources.common import DEFAULT_MODEL_DIR, load_resources_json
    os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)
    return load_resources_json(DEFAULT_MODEL_DIR)


def find_stanza_match(ud_lang_code, package_hint, treebank_dir, stanza_index):
    """
    Map a UD language code to a Stanza language code.
    Returns the Stanza lang string, or None if no usable model exists.

    We do NOT pin a specific package -- Stanza's per-processor defaults
    handle the case where (e.g.) a heavyweight `combined_camembert-large`
    package only ships POS+depparse models, while `tokenize` and `lemma`
    live under separate per-treebank packages. Letting Stanza resolve
    each processor independently is what `stanza.Pipeline('fr')` does
    out of the box, and it Just Works for every language.
    """
    # Codes that diverge between UD and Stanza
    if ud_lang_code == "zh":
        # UD_Chinese-GSD = traditional, UD_Chinese-GSDSimp = simplified
        candidates = ["zh-hans"] if "Simp" in treebank_dir else ["zh-hant"]
    elif ud_lang_code == "no":
        if "bokmaal" in package_hint:
            candidates = ["nb"]
        elif "nynorsk" in package_hint:
            candidates = ["nn"]
        else:
            candidates = ["nb", "nn"]
    else:
        candidates = [ud_lang_code]

    for slang in candidates:
        if slang not in stanza_index:
            continue
        entry = stanza_index[slang]
        # Need tokenize, pos, and depparse to exist *somewhere* for this
        # language. (POSProcessor declares tokenize as a hard requirement
        # even when tokenize_pretokenized=True is set later.) Lemma is
        # optional -- Stanza falls back to identity, and we use gold
        # lemmas downstream anyway.
        if (entry.get("tokenize")
                and entry.get("pos")
                and entry.get("depparse")):
            return slang
    return None


def discover_treebanks(ud_root, min_tokens=MIN_TOKENS):
    """
    Walk UD treebanks under ud_root and return specs for the eligible ones,
    keeping only the largest per Stanza language.
    """
    print("Loading Stanza resource index...")
    stanza_index = load_stanza_index()
    print(f"  Stanza supports {len(stanza_index)} languages")

    candidates = []
    skipped    = []
    n_dirs     = 0

    for entry in sorted(os.listdir(ud_root)):
        if not entry.startswith("UD_"):
            continue
        tb_dir = os.path.join(ud_root, entry)
        if not os.path.isdir(tb_dir):
            continue
        n_dirs += 1

        files = os.listdir(tb_dir)
        train_files = [f for f in files if f.endswith("-ud-train.conllu")]
        test_files  = [f for f in files if f.endswith("-ud-test.conllu")]
        if not train_files or not test_files:
            skipped.append((entry, "no train+test conllu"))
            continue
        train_file = train_files[0]
        test_file  = test_files[0]

        prefix = train_file.replace("-ud-train.conllu", "")
        if "_" not in prefix:
            skipped.append((entry, f"unrecognised prefix: {prefix}"))
            continue
        lang_code, package = prefix.split("_", 1)

        slang = find_stanza_match(lang_code, package, entry, stanza_index)
        if not slang:
            skipped.append((entry, f"no Stanza model for {lang_code}"))
            continue

        train_path = os.path.join(tb_dir, train_file)
        if not has_lemmas(train_path):
            skipped.append((entry, "no lemmas annotated"))
            continue

        n_train = count_tokens(train_path)
        n_test  = count_tokens(os.path.join(tb_dir, test_file))
        n_total = n_train + n_test
        if n_total < min_tokens:
            skipped.append((entry, f"too small ({n_total} tokens)"))
            continue

        # UD_French-GSD -> French
        lang_full = entry[len("UD_"):].rsplit("-", 1)[0]

        candidates.append({
            "name":        prefix,
            "language":    lang_full,
            "ud_lang":     lang_code,
            "ud_package":  package,
            "path":        tb_dir,
            "train_file":  train_file,
            "test_file":   test_file,
            "stanza_lang": slang,
            "n_train":     n_train,
            "n_test":      n_test,
            "n_tokens":    n_total,
        })

    # Dedup by Stanza language: keeps Norwegian Bokmål and Nynorsk separate
    # (since they have distinct parsers), but folds e.g. UD English EWT, GUM,
    # ParTUT into a single English entry.
    best = {}
    for c in candidates:
        key = c["stanza_lang"]
        cur = best.get(key)
        if cur is None or c["n_tokens"] > cur["n_tokens"]:
            best[key] = c

    selected = sorted(best.values(), key=lambda c: -c["n_tokens"])

    print(f"\nDiscovery summary:")
    print(f"  scanned        {n_dirs} UD_* directories")
    print(f"  passed checks  {len(candidates)} treebanks")
    print(f"  after dedup    {len(selected)} treebanks (largest per language)")
    print(f"  skipped        {len(skipped)} treebanks")
    if skipped:
        reason_counts = Counter(r for _, r in skipped)
        print("\nSkip reasons:")
        for r, n in reason_counts.most_common():
            print(f"  {n:4d}  {r}")

    return selected


# ---------------------------------------------------------------------------
# Per-treebank processing
# ---------------------------------------------------------------------------

def compute_train_scores(spec):
    """Run the stability scoring pipeline on the train portion only."""
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
    try:
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
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    token_freqs = Counter()
    with open(train_path, encoding="utf-8") as f:
        for sent in conllu.parse_incr(f):
            for tok in sent:
                if isinstance(tok["id"], int):
                    token_freqs[(tok["lemma"], tok["upos"])] += 1
    return scores, token_freqs


def run_parser(spec):
    """Parse the test set with Stanza using gold tokenization."""
    import stanza
    lang = spec["stanza_lang"]
    procs = "tokenize,pos,lemma,depparse"
    # No `package=` kwarg: let Stanza resolve a default package per
    # processor independently. This is the same behaviour you get from
    # `stanza.Pipeline('fr')` with no extra config, and it sidesteps the
    # case where heavyweight packages (e.g. combined_camembert-large)
    # only ship POS+depparse models and would silently drop tokenize.
    stanza.download(lang, processors=procs, verbose=False)
    nlp = stanza.Pipeline(
        lang=lang,
        processors=procs,
        tokenize_pretokenized=True,
        use_gpu=False,
        verbose=False,
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


def process_treebank(spec, cache_dir, force=False):
    print(f"\n{'='*72}")
    print(f"{spec['name']}  ({spec['language']}, {spec['n_tokens']:,} tokens, "
          f"stanza={spec['stanza_lang']})")
    print(f"{'='*72}")

    tb_cache = os.path.join(cache_dir, spec["name"])
    os.makedirs(tb_cache, exist_ok=True)
    with open(os.path.join(tb_cache, "spec.json"), "w") as f:
        json.dump(spec, f, indent=2)

    # ---- Step 1: train stability scores ----------------------------------
    scores_pkl = os.path.join(tb_cache, "scores.pkl")
    freqs_pkl  = os.path.join(tb_cache, "freqs.pkl")
    if force or not (os.path.exists(scores_pkl) and os.path.exists(freqs_pkl)):
        print("  [1/3] Computing train stability scores...")
        scores, freqs = compute_train_scores(spec)
        with open(scores_pkl, "wb") as f: pickle.dump(scores, f)
        with open(freqs_pkl, "wb") as f: pickle.dump(freqs, f)
        print(f"        {len(scores):,} lex units scored")
    else:
        print("  [1/3] cached -> loading train stability scores")
        with open(scores_pkl, "rb") as f: scores = pickle.load(f)
        with open(freqs_pkl, "rb") as f: freqs = pickle.load(f)

    # ---- Step 2: parser --------------------------------------------------
    parsed_pkl = os.path.join(tb_cache, "parsed.pkl")
    if force or not os.path.exists(parsed_pkl):
        print("  [2/3] Running Stanza parser on test set...")
        parsed = run_parser(spec)
        with open(parsed_pkl, "wb") as f: pickle.dump(parsed, f)
        print(f"        {sum(len(s) for s in parsed):,} test tokens parsed")
    else:
        print("  [2/3] cached -> loading parser output")
        with open(parsed_pkl, "rb") as f: parsed = pickle.load(f)

    # ---- Step 3: joined dataframe ----------------------------------------
    print("  [3/3] Building joined dataframe...")
    df = build_dataframe(parsed, scores, freqs)
    df.to_csv(os.path.join(tb_cache, "tokens.csv"), index=False)
    n_err = int(df["pos_err"].sum())
    n_tok = len(df)
    err_rate = n_err / n_tok if n_tok else 0.0
    print(f"        {n_tok:,} tokens, {n_err:,} POS errors ({err_rate:.2%})")


# ---------------------------------------------------------------------------
# Analysis (works off cached CSVs only)
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


def fit_regression(df):
    sub = df.dropna(subset=["purity"]).copy()
    if len(sub) < 50 or sub["pos_err"].sum() < 5:
        return None
    try:
        m  = smf.logit("pos_err ~ purity + log_freq", data=sub).fit(disp=0)
        ci = m.conf_int().loc["purity"]
        return {
            "n":             len(sub),
            "n_errors":      int(sub["pos_err"].sum()),
            "coef":          float(m.params["purity"]),
            "se":            float(m.bse["purity"]),
            "ci_low":        float(ci.iloc[0]),
            "ci_high":       float(ci.iloc[1]),
            "p":             float(m.pvalues["purity"]),
            "log_freq_coef": float(m.params["log_freq"]),
        }
    except Exception as e:
        print(f"    regression failed: {e}")
        return None


def rival_alignment(df, threshold=RIVAL_PROB_THRESHOLD):
    """
    For each POS error, check whether pred_upos is among the lex unit's
    *meaningful rivals* -- every category with normalized mass >= threshold,
    excluding the gold category itself.
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

    pur_med = sub["purity"].median()
    ent_med = sub["entropy"].median()
    sub["pur_lvl"] = np.where(sub["purity"]  <  pur_med, "low_pur",  "high_pur")
    sub["ent_lvl"] = np.where(sub["entropy"] >= ent_med, "high_ent", "low_ent")
    sub["cell"]    = sub["pur_lvl"] + " x " + sub["ent_lvl"]

    errs = sub[sub["pos_err"] == 1]
    if len(errs) == 0:
        return {"n_errors": 0, "overall_hit_rate": np.nan}

    out = {"n_errors": int(len(errs)),
           "overall_hit_rate": float(errs["pred_in_rivals"].mean())}
    for cell in ["low_pur x low_ent",  "low_pur x high_ent",
                 "high_pur x low_ent", "high_pur x high_ent"]:
        ce = errs[errs["cell"] == cell]
        out[f"hit_{cell}"] = (float(ce["pred_in_rivals"].mean())
                              if len(ce) else np.nan)
        out[f"n_{cell}"]   = int(len(ce))
    return out


def make_forest_plot(summary_df, out_path):
    valid = summary_df.dropna(subset=["coef"]).sort_values("coef")
    if len(valid) == 0:
        print("No valid regressions to plot.")
        return
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(valid))))
    y = np.arange(len(valid))
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
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved forest plot: {out_path}")


def analyze(cache_dir, output_dir):
    """Read cached per-treebank CSVs, fit regressions, build summary + plot."""
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.isdir(cache_dir):
        print(f"No cache directory at {cache_dir}.")
        return

    summary = []
    for name in sorted(os.listdir(cache_dir)):
        tb_cache = os.path.join(cache_dir, name)
        csv_path = os.path.join(tb_cache, "tokens.csv")
        if not os.path.exists(csv_path):
            continue
        print(f"\n--- {name} ---")
        df = pd.read_csv(csv_path)
        reg = fit_regression(df)
        ra  = rival_alignment(df)
        row = {"treebank":      name,
               "n_test_tokens": len(df),
               "n_pos_errors":  int(df["pos_err"].sum())}
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

    if not summary:
        print("No cached treebanks found in", cache_dir)
        return

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))

    make_forest_plot(summary_df, os.path.join(output_dir, "forest_plot.png"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ud-root",     default=UD_ROOT,
                   help="Directory containing all UD_* treebanks.")
    p.add_argument("--cache-dir",   default=CACHE_DIR,
                   help="Where per-treebank intermediates are saved.")
    p.add_argument("--output-dir",  default=OUTPUT_DIR,
                   help="Where summary.csv and plots are written.")
    p.add_argument("--min-tokens",  type=int, default=MIN_TOKENS,
                   help="Minimum total tokens (train+test) for a treebank.")
    p.add_argument("--force",       action="store_true",
                   help="Recompute even if cached artefacts exist.")
    p.add_argument("--analyze-only", action="store_true",
                   help="Skip data collection; just rebuild summary + plots.")
    p.add_argument("--dry-run",     action="store_true",
                   help="Print eligible treebanks and exit.")
    p.add_argument("--only", nargs="+", default=None,
                   help="Restrict to these treebank names (e.g. fr_gsd en_ewt).")
    return p.parse_args()


def main():
    args = parse_args()

    if args.analyze_only:
        analyze(args.cache_dir, args.output_dir)
        return

    specs = discover_treebanks(args.ud_root, args.min_tokens)
    if args.only:
        wanted = set(args.only)
        specs  = [s for s in specs if s["name"] in wanted]

    print(f"\nSelected {len(specs)} treebank(s):")
    print(f"  {'name':<22} {'language':<22} {'tokens':>10}  stanza")
    for s in specs:
        print(f"  {s['name']:<22} {s['language']:<22} {s['n_tokens']:>10,}  "
              f"{s['stanza_lang']}")

    if args.dry_run:
        return

    for spec in specs:
        try:
            process_treebank(spec, args.cache_dir, force=args.force)
        except Exception:
            traceback.print_exc()

    analyze(args.cache_dir, args.output_dir)


if __name__ == "__main__":
    main()