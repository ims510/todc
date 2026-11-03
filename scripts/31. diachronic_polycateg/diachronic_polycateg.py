"""
diachronic_polycategoriality.py
===============================
Diachronic extension of the stability / intruder pipeline.

It reuses the metric code from your existing module
(`stability_intruder_matrix_claude.py`) unchanged, and adds:

  PASS 1 (heavy, cached per year)
    - stability & intruder matrices            -> <out>/<year>/stability.csv, intruder.csv
    - per-unit records (purity, entropy, rival,
      and the numeric p_norm over categories)  -> <out>/<year>/units.csv
    - the aligned feature matrix + index maps   -> <out>/<year>/profiles.npz
    The corpus and the n^2 similarity matrix are dropped before the next year,
    so peak memory stays at one year at a time.

  PASS 2 (cheap, cross-year, frame-aware)
    - stability lines (category cohesion over time)
    - chosen intruder cells over time (e.g. NOUN->ADP)
    - per-word neighbourhood-composition stacked area
    - per-word purity/entropy trajectory in the 2D state space
    - automatic "top movers" ranking (distribution drift + profile drift)
    - per-word feature delta (early window vs late window) -> which features moved

Design notes
------------
* Two signals are kept separate (see the stacked-area / feature-delta vs the
  purity-entropy views): the *intrinsic* profile is frame-free, the *relational*
  numbers (purity/entropy/p_norm) are read "given that year's space".
* All cross-year vector comparisons align to the UNION feature vocabulary,
  reindexed with 0 for absent features. POS/category sets are likewise reindexed.
* Because the same Stanza model parses every year, parser bias is ~constant:
  treat differences as diachronic signal, not absolute values comparable to GSD.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import entropy as shannon_entropy

# --- reuse your existing pipeline unchanged -------------------------------
import sys
# Make sure stability_intruder_matrix_claude.py is importable (same folder or PYTHONPATH).
import stability_intruder_matrix_claude as base
sys.path.insert(1, "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/tod")
import tod.corpus

# ===========================================================================
# CONFIG
# ===========================================================================
TRIAD_ROOT = os.path.join(os.path.dirname(__file__), "TRIAD")
TEXT_DIR = os.path.join(TRIAD_ROOT, "Textes_tagged_stanza")
METADATA = os.path.join(TRIAD_ROOT, "metadata.tsv")
YEAR_TREEBANK_DIR = os.path.join(TRIAD_ROOT, "by_year")
YEARS = [(y, f"/path/to/corpora/{y}.conllu") for y in range(1984, 2019)]

OUT_DIR = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/31. diachronic_polycateg/output"

# Words to track in detail (lemma, pos). Add your hypotheses here.
WATCHLIST = [
    ("cadre", "NOUN"),
    ("niveau", "NOUN"),
    ("partir", "VERB"),   # "à partir de" -> relator drift, same family as your cadre case
]

# Intruder cells to follow over time, as (original_category, rival_category).
INTRUDER_CELLS = [
    ("NOUN", "ADP"),
    ("ADJ", "VERB"),
    ("PRON", "ADV"),
]

# Mover-ranking parameters
MIN_YEARS_PRESENT = 10   # a unit must appear in at least this many years to be ranked
WINDOW = 3               # size of the early / late windows used for drift & deltas


# ===========================================================================
# PASS 1 helpers
# ===========================================================================

def build_year_groups(metadata_path=METADATA, text_dir=TEXT_DIR):
    df = pd.read_csv(metadata_path, sep="\t")
    if "annee" not in df.columns:
        df.columns = ["name", "target", "decennie", "annee"]

    df["path"] = df["name"].apply(lambda name: os.path.join(text_dir, f"{name}.conllu"))
    grouped = df.groupby("annee")["path"].apply(list)
    grouped.to_csv("/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/31. diachronic_polycateg/output.csv")
    
    return [(int(year), paths) for year, paths in sorted(grouped.items())]

import shutil

def prepare_year_dirs(year_groups, yearbank_root=YEAR_TREEBANK_DIR):
    os.makedirs(yearbank_root, exist_ok=True)
    out = []
    for year, paths in year_groups:
        ydir = os.path.join(yearbank_root, str(year))
        os.makedirs(ydir, exist_ok=True)
        for p in paths:
            shutil.copy(p, ydir)
        out.append((year, ydir))
    return out

YEARS = prepare_year_dirs(build_year_groups())

def load_year_corpus(path):
    """Mirror base.load_corpus() but for an arbitrary path (one year)."""
    return tod.corpus.Corpus(
        treebank_path=path,
        grew_pattern=base.GREW_PATTERN,
        patterns_text_file=base.PATTERNS_TEXT_FILE,
        use_sud=False,
        matrix_type="coverage",
        excluded_feature_patterns=base.EXCLUDED_FEATURE_PATTERNS,
        included_feature_patterns=base.INCLUDED_FEATURE_PATTERNS,
    )


def per_unit_records(corpus, similarity_matrix, category_freqs):
    """
    Like base.compute_scores(), but keeps the NUMERIC p_norm distribution
    (one column per category) instead of a display string. Reuses base helpers.
    """
    records = []
    for idx, lex_unit in corpus._idx2lexunit.items():
        current_cat = lex_unit[1]
        neighbours = base._get_neighbours(idx, similarity_matrix)
        if not neighbours:
            continue

        raw_weights, total_weight = {}, 0.0
        for n_idx in neighbours:
            n_cat = corpus._idx2lexunit[n_idx][1]
            sim = float(similarity_matrix[idx][n_idx])
            raw_weights[n_cat] = raw_weights.get(n_cat, 0.0) + sim
            total_weight += sim

        p_norm = base._freq_normalised_probs(raw_weights, total_weight, category_freqs)
        purity = p_norm.get(current_cat, 0.0)
        ent = float(shannon_entropy(list(p_norm.values()), base=2))
        rival_cat, rival_p = None, 0.0
        for cat, p in p_norm.items():
            if cat != current_cat and p > rival_p:
                rival_cat, rival_p = cat, p

        row = {
            "lemma": lex_unit[0],
            "pos": current_cat,
            "purity": purity,
            "entropy": ent,
            "rival": rival_cat,
            "rival_p": rival_p,
            "n_neighbours": len(neighbours),
        }
        for cat, p in p_norm.items():
            row[f"p_{cat}"] = p          # numeric neighbourhood make-up
        records.append(row)
    return pd.DataFrame(records)


def run_pass1(years=YEARS, out_dir=OUT_DIR):
    """Process each year once; cache everything cross-year analysis needs."""
    os.makedirs(out_dir, exist_ok=True)
    size_report = []

    for year, path in years:
        ydir = os.path.join(out_dir, str(year))
        os.makedirs(ydir, exist_ok=True)
        print(f"[pass1] {year} ...")

        corpus = load_year_corpus(path)
        sim = base.build_similarity_matrix(corpus)
        cat_freqs = base._global_category_freqs(corpus)

        stability, intruder = base.build_matrices(corpus, sim, excluded_tags=base.EXCLUDED_TAGS)
        stability.to_csv(os.path.join(ydir, "stability.csv"))
        intruder.to_csv(os.path.join(ydir, "intruder.csv"))

        units = per_unit_records(corpus, sim, cat_freqs)
        units.to_csv(os.path.join(ydir, "units.csv"), index=False)

        # Aligned feature matrix + index maps (name-keyed so Pass 2 can reindex).
        feature_names = [corpus._idx2feature[i] for i in range(len(corpus._idx2feature))]
        lemmas = [corpus._idx2lexunit[i][0] for i in range(len(corpus._idx2lexunit))]
        poss   = [corpus._idx2lexunit[i][1] for i in range(len(corpus._idx2lexunit))]
        np.savez_compressed(
            os.path.join(ydir, "profiles.npz"),
            matrix=np.asarray(corpus.feature_matrix, dtype=np.float32),
            feature_names=np.array(feature_names, dtype=object),
            lemmas=np.array(lemmas, dtype=object),
            poss=np.array(poss, dtype=object),
        )

        n_tokens = getattr(corpus, "n_tokens", None)
        size_report.append({"year": year, "n_units": len(units),
                            "n_features": len(feature_names), "n_tokens": n_tokens})

        # free the heavy objects before the next year
        del corpus, sim
        print(f"[pass1] {year} done ({len(units)} units, {len(feature_names)} features)")

    pd.DataFrame(size_report).to_csv(os.path.join(out_dir, "size_report.csv"), index=False)
    print("[pass1] size_report.csv written -- check for large year-to-year size swings.")


# ===========================================================================
# PASS 2 loaders
# ===========================================================================

def _years_present(out_dir):
    return sorted(int(d) for d in os.listdir(out_dir)
                  if d.isdigit() and os.path.isdir(os.path.join(out_dir, d)))


def load_all_units(out_dir=OUT_DIR):
    """Concatenate every year's per-unit records into one tidy long table."""
    frames = []
    for y in _years_present(out_dir):
        df = pd.read_csv(os.path.join(out_dir, str(y), "units.csv"))
        df["year"] = y
        frames.append(df)
    allu = pd.concat(frames, ignore_index=True)
    # category columns (p_*) reindexed to the union, absent -> 0
    pcols = sorted(c for c in allu.columns if c.startswith("p_"))
    allu[pcols] = allu[pcols].fillna(0.0)
    return allu, pcols


def load_matrix(out_dir, year, kind):
    """kind in {'stability','intruder'}."""
    return pd.read_csv(os.path.join(out_dir, str(year), f"{kind}.csv"), index_col=0)


def load_profile_vector(out_dir, year, lemma, pos):
    """Return {feature_name: value} for one (lemma,pos) in one year, or None."""
    npz = np.load(os.path.join(out_dir, str(year), "profiles.npz"), allow_pickle=True)
    lemmas, poss = npz["lemmas"], npz["poss"]
    mask = (lemmas == lemma) & (poss == pos)
    if not mask.any():
        return None
    row = npz["matrix"][np.argmax(mask)]
    names = npz["feature_names"]
    return {names[i]: float(row[i]) for i in range(len(names)) if row[i] != 0.0}


# ===========================================================================
# PASS 2 views
# ===========================================================================

def plot_stability_lines(out_dir=OUT_DIR, save=None):
    years = _years_present(out_dir)
    diag = {}
    for y in years:
        m = load_matrix(out_dir, y, "stability")
        diag[y] = pd.Series(np.diag(m.values), index=m.index)
    df = pd.DataFrame(diag).T  # rows=years, cols=categories
    plt.figure(figsize=(11, 6))
    for cat in df.columns:
        plt.plot(df.index, df[cat], marker="o", ms=3, label=cat)
    plt.xlabel("Year"); plt.ylabel("Stability (internal cohesion)")
    plt.title("Category stability over time")
    plt.legend(ncol=4, fontsize=8); plt.tight_layout()
    if save: plt.savefig(save, dpi=150)
    plt.show(); plt.close()
    return df


def plot_intruder_cells(cells=INTRUDER_CELLS, out_dir=OUT_DIR, save=None):
    years = _years_present(out_dir)
    series = {f"{o}->{r}": [] for o, r in cells}
    for y in years:
        m = load_matrix(out_dir, y, "intruder")
        for o, r in cells:
            v = m.loc[o, r] if (o in m.index and r in m.columns) else np.nan
            series[f"{o}->{r}"].append(v)
    plt.figure(figsize=(11, 6))
    for k, vals in series.items():
        plt.plot(years, vals, marker="o", ms=3, label=k)
    plt.xlabel("Year"); plt.ylabel("Intruder weight (renormalised off-diagonal)")
    plt.title("Selected categorical overlaps over time")
    plt.legend(); plt.tight_layout()
    if save: plt.savefig(save, dpi=150)
    plt.show(); plt.close()
    return pd.DataFrame(series, index=years)


def plot_neighbourhood_area(lemma, pos, allu=None, pcols=None,
                            out_dir=OUT_DIR, save=None):
    """Stacked area: how the word's neighbourhood make-up redistributes by year."""
    if allu is None:
        allu, pcols = load_all_units(out_dir)
    sub = allu[(allu.lemma == lemma) & (allu.pos == pos)].sort_values("year")
    if sub.empty:
        print(f"{lemma}/{pos} never clears the threshold."); return None
    # keep only categories that ever carry non-trivial mass, for legibility
    keep = [c for c in pcols if sub[c].max() > 0.03]
    data = sub.set_index("year")[keep]
    plt.figure(figsize=(11, 6))
    plt.stackplot(data.index, data.T.values, labels=[c[2:] for c in keep])
    plt.xlabel("Year"); plt.ylabel("Neighbourhood proportion (p_norm)")
    plt.title(f"Neighbourhood composition: {lemma} / {pos}")
    plt.legend(ncol=4, fontsize=8, loc="upper left"); plt.tight_layout()
    if save: plt.savefig(save, dpi=150)
    plt.show(); plt.close()
    return data


def plot_purity_entropy_trajectory(lemma, pos, allu=None, out_dir=OUT_DIR, save=None):
    if allu is None:
        allu, _ = load_all_units(out_dir)
    sub = allu[(allu.lemma == lemma) & (allu.pos == pos)].sort_values("year")
    if sub.empty:
        print(f"{lemma}/{pos} never clears the threshold."); return None
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(sub.purity, sub.entropy, c=sub.year, cmap="viridis", s=40, zorder=3)
    plt.plot(sub.purity, sub.entropy, "-", alpha=0.4, zorder=2)
    plt.colorbar(sc, label="Year")
    plt.xlabel("Purity"); plt.ylabel("Entropy")
    plt.title(f"State-space trajectory: {lemma} / {pos}\n(low purity, low entropy = focused bridge)")
    plt.tight_layout()
    if save: plt.savefig(save, dpi=150)
    plt.show(); plt.close()
    return sub[["year", "purity", "entropy", "rival"]]


def top_movers(allu=None, pcols=None, out_dir=OUT_DIR,
               min_years=MIN_YEARS_PRESENT, window=WINDOW, top_n=30):
    """
    Rank units by how much they moved. Two scores:
      dist_drift    - total-variation distance between early-window and late-window
                      mean neighbourhood make-up (changed category allegiance)
      purity_change - late mean purity minus early mean purity (signed)
    """
    if allu is None:
        allu, pcols = load_all_units(out_dir)
    rows = []
    for (lemma, pos), g in allu.groupby(["lemma", "pos"]):
        g = g.sort_values("year")
        if g.year.nunique() < min_years:
            continue
        early = g.head(window); late = g.tail(window)
        pe = early[pcols].mean().values
        pl = late[pcols].mean().values
        tvd = 0.5 * np.abs(pe - pl).sum()
        rows.append({
            "lemma": lemma, "pos": pos, "n_years": g.year.nunique(),
            "dist_drift": tvd,
            "purity_change": late.purity.mean() - early.purity.mean(),
            "entropy_change": late.entropy.mean() - early.entropy.mean(),
            "early_rival": early.rival.mode().iat[0] if not early.rival.mode().empty else None,
            "late_rival": late.rival.mode().iat[0] if not late.rival.mode().empty else None,
        })
    out = pd.DataFrame(rows).sort_values("dist_drift", ascending=False)
    return out.head(top_n).reset_index(drop=True)


def feature_delta(lemma, pos, out_dir=OUT_DIR, window=WINDOW, top_k=25):
    """
    Frame-free diagnosis: average the word's aligned profile over its first
    `window` years of presence and its last `window`, then rank features by
    absolute change. This is what surfaces e.g. prev:upos=ADP rising for `cadre`.
    """
    years = _years_present(out_dir)
    present = []
    for y in years:
        v = load_profile_vector(out_dir, y, lemma, pos)
        if v is not None:
            present.append((y, v))
    if len(present) < 2:
        print(f"{lemma}/{pos} present in <2 years; cannot delta."); return None

    early = present[:window]; late = present[-window:]
    feats = set().union(*[v for _, v in early], *[v for _, v in late])

    def avg(window_items):
        return {f: np.mean([v.get(f, 0.0) for _, v in window_items]) for f in feats}

    e, l = avg(early), avg(late)
    df = pd.DataFrame({
        "feature": list(feats),
        "early": [e[f] for f in feats],
        "late":  [l[f] for f in feats],
    })
    df["delta"] = df["late"] - df["early"]
    df["abs_delta"] = df["delta"].abs()
    return df.sort_values("abs_delta", ascending=False).head(top_k).reset_index(drop=True)


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    # 1) cache everything (run once; comment out afterwards)
    run_pass1()

    # 2) cross-year analysis
    allu, pcols = load_all_units()
    plot_stability_lines(save=os.path.join(OUT_DIR, "stability_lines.png"))
    plot_intruder_cells(save=os.path.join(OUT_DIR, "intruder_cells.png"))

    movers = top_movers(allu, pcols)
    movers.to_csv(os.path.join(OUT_DIR, "top_movers.csv"), index=False)
    print("\nTop movers by neighbourhood-distribution drift:\n", movers.head(15).to_string(index=False))

    for lemma, pos in WATCHLIST:
        plot_neighbourhood_area(lemma, pos, allu, pcols,
                                save=os.path.join(OUT_DIR, f"area_{lemma}_{pos}.png"))
        plot_purity_entropy_trajectory(lemma, pos, allu,
                                save=os.path.join(OUT_DIR, f"traj_{lemma}_{pos}.png"))
        fd = feature_delta(lemma, pos)
        if fd is not None:
            print(f"\nFeature delta {lemma}/{pos} (early vs late):\n", fd.to_string(index=False))