"""
diachronic_polycateg_v2.py
==========================
Revision of the diachronic pipeline. Changes vs the previous version:

  * BINNING: process one bin per *decade* (or per year), pooling files via the
    metadata.tsv `decennie` / `annee` columns. Pooling cures the degenerate
    single-neighbour estimates and the size-imbalance artifact in the stability
    plot. Set BIN_BY = "decennie" or "annee".

  * DISCOVERY: `discover_movers()` ranks candidates by FRAME-FREE feature-profile
    drift (L1 distance between first-bin and last-bin profiles), not by the noisy
    neighbourhood TVD. Each candidate is annotated with the category centroid it
    drifted *toward*, and flagged if that target is a functional category
    (content -> function = the grammaticalisation signature, e.g. partir).
    Filters: content POS only, present in >= MIN_BINS bins.

  * PLOTS: neighbourhood composition is now LINES (one per category) with integer
    ticks; the trajectory is unlinked points annotated with the bin label.

  * Run the SOURCE/DECADE confound check before trusting cross-decade results
    (see the docstring of source_decade_crosstab).
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import entropy as shannon_entropy

import sys
import stability_intruder_matrix_claude as base
sys.path.insert(1, "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/tod")
import tod.corpus

# ===========================================================================
# CONFIG
# ===========================================================================
TRIAD_ROOT = os.path.join(os.path.dirname(__file__), "TRIAD")
TEXT_DIR   = os.path.join(TRIAD_ROOT, "Textes_tagged_stanza")
METADATA   = os.path.join(TRIAD_ROOT, "metadata.tsv")
BIN_DIR    = os.path.join(TRIAD_ROOT, "by_bin")

BIN_BY = "decennie"          # "decennie" (4 bins) or "annee" (yearly)
OUT_DIR    = os.path.join("/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/31. diachronic_polycateg/output/", BIN_BY)

# Discovery filters
CONTENT_POS    = {"NOUN", "VERB", "ADJ", "ADV"}
FUNCTIONAL_POS = {"ADP", "DET", "SCONJ", "CCONJ", "PART", "AUX", "PRON"}
EXCLUDED_TAGS  = ["SYM", "X"]   # dropped from the stability / intruder matrices
MIN_BINS       = 3              # candidate must appear in at least this many bins

WATCHLIST = [("cadre", "NOUN"), ("niveau", "NOUN"), ("partir", "VERB"),
             ("compte", "NOUN")]
INTRUDER_CELLS = [("NOUN", "ADP"), ("ADJ", "VERB"), ("PRON", "ADV")]


# ===========================================================================
# Confound check  (run this once, look at it, before interpreting anything)
# ===========================================================================

def source_decade_crosstab(metadata_path=METADATA):
    """
    Your filenames look like two archives (fradhds-... early, CA-Nanterre-...
    later). If source and decade are confounded, cross-decade 'drift' may be a
    source/OCR artifact. If the sources overlap in time you can adjust for it;
    if they partition the timeline, restrict your strongest claims to the span
    covered by a single source.
    """
    df = pd.read_csv(metadata_path, sep="\t")
    df["source"] = df["name"].str.split(r"[-_]").str[0]
    ct = pd.crosstab(df["source"], df["decennie"])
    print("Source x decade:\n", ct, "\n")
    return ct


# ===========================================================================
# BINNING
# ===========================================================================

def build_bin_groups(metadata_path=METADATA, text_dir=TEXT_DIR, by=BIN_BY):
    df = pd.read_csv(metadata_path, sep="\t")
    df["path"] = df["name"].apply(lambda n: os.path.join(text_dir, f"{n}.conllu"))
    grouped = df.groupby(by)["path"].apply(list)
    return [(int(label), paths) for label, paths in sorted(grouped.items())]


def prepare_bin_dirs(bin_groups, root=BIN_DIR):
    os.makedirs(root, exist_ok=True)
    out = []
    for label, paths in bin_groups:
        bdir = os.path.join(root, str(label))
        os.makedirs(bdir, exist_ok=True)
        for p in paths:
            if os.path.exists(p):
                shutil.copy(p, bdir)
        out.append((label, bdir))
    return out


def load_bin_corpus(path):
    return tod.corpus.Corpus(
        treebank_path=path,
        grew_pattern=base.GREW_PATTERN,
        patterns_text_file=base.PATTERNS_TEXT_FILE,
        use_sud=False,
        matrix_type="coverage",
        excluded_feature_patterns=base.EXCLUDED_FEATURE_PATTERNS,
        included_feature_patterns=base.INCLUDED_FEATURE_PATTERNS,
    )


# ===========================================================================
# PASS 1
# ===========================================================================

def per_unit_records(corpus, similarity_matrix, category_freqs):
    """Per-lexical-unit purity/entropy/rival + numeric p_norm over categories."""
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
        row = {"lemma": lex_unit[0], "pos": current_cat, "purity": purity,
               "entropy": ent, "rival": rival_cat, "rival_p": rival_p,
               "n_neighbours": len(neighbours)}
        for cat, p in p_norm.items():
            row[f"p_{cat}"] = p
        records.append(row)
    return pd.DataFrame(records)


def run_pass1(bins, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    size_report = []
    for label, path in bins:
        bdir = os.path.join(out_dir, str(label))
        os.makedirs(bdir, exist_ok=True)
        print(f"[pass1] bin {label} ...")
        corpus = load_bin_corpus(path)
        sim = base.build_similarity_matrix(corpus)
        cat_freqs = base._global_category_freqs(corpus)

        stability, intruder = base.build_matrices(corpus, sim, excluded_tags=EXCLUDED_TAGS)
        stability.to_csv(os.path.join(bdir, "stability.csv"))
        intruder.to_csv(os.path.join(bdir, "intruder.csv"))
        per_unit_records(corpus, sim, cat_freqs).to_csv(
            os.path.join(bdir, "units.csv"), index=False)

        feature_names = [corpus._idx2feature[i] for i in range(len(corpus._idx2feature))]
        lemmas = [corpus._idx2lexunit[i][0] for i in range(len(corpus._idx2lexunit))]
        poss   = [corpus._idx2lexunit[i][1] for i in range(len(corpus._idx2lexunit))]
        np.savez_compressed(
            os.path.join(bdir, "profiles.npz"),
            matrix=np.asarray(corpus.feature_matrix, dtype=np.float32),
            feature_names=np.array(feature_names, dtype=object),
            lemmas=np.array(lemmas, dtype=object),
            poss=np.array(poss, dtype=object))

        size_report.append({"bin": label, "n_units": len(lemmas),
                             "n_features": len(feature_names)})
        del corpus, sim
        print(f"[pass1] bin {label} done ({len(lemmas)} units)")
    pd.DataFrame(size_report).to_csv(os.path.join(out_dir, "size_report.csv"), index=False)


# ===========================================================================
# PASS 2 loaders
# ===========================================================================

def _bins_present(out_dir=OUT_DIR):
    return sorted(int(d) for d in os.listdir(out_dir)
                  if d.isdigit() and os.path.isdir(os.path.join(out_dir, d)))


def load_all_units(out_dir=OUT_DIR):
    frames = []
    for b in _bins_present(out_dir):
        df = pd.read_csv(os.path.join(out_dir, str(b), "units.csv"))
        df["bin"] = b
        frames.append(df)
    allu = pd.concat(frames, ignore_index=True)
    pcols = sorted(c for c in allu.columns if c.startswith("p_"))
    allu[pcols] = allu[pcols].fillna(0.0)
    return allu, pcols


def load_matrix(out_dir, b, kind):
    return pd.read_csv(os.path.join(out_dir, str(b), f"{kind}.csv"), index_col=0)


def load_bin_profile_df(out_dir, b):
    """Profiles of one bin as a DataFrame indexed by (lemma, pos), cols = features."""
    npz = np.load(os.path.join(out_dir, str(b), "profiles.npz"), allow_pickle=True)
    idx = pd.MultiIndex.from_tuples(list(zip(npz["lemmas"], npz["poss"])),
                                    names=["lemma", "pos"])
    df = pd.DataFrame(npz["matrix"], columns=list(npz["feature_names"]), index=idx)
    return df[~df.index.duplicated(keep="first")]


def _cos(s1, s2):
    feats = s1.index.union(s2.index)
    a = s1.reindex(feats, fill_value=0.0).values
    b = s2.reindex(feats, fill_value=0.0).values
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


# ===========================================================================
# DISCOVERY  (frame-free feature drift + directional annotation)
# ===========================================================================

def discover_movers(out_dir=OUT_DIR, content_pos=CONTENT_POS,
                    functional_pos=FUNCTIONAL_POS, min_bins=MIN_BINS, top_n=50):
    """
    Rank content words by how much their raw syntactic profile changed between
    their first and last bin, then annotate the direction of change.

    Columns
    -------
    drift        : 0.5 * L1( first_profile, last_profile )  (frame-free)
    toward       : category centroid the profile moved closest to (excl. own)
    functional   : True if `toward` is a functional category (grammaticalisation)
    sim_gain     : increase in cosine similarity to that centroid (first->last)
    """
    bins = _bins_present(out_dir)
    dfs = {b: load_bin_profile_df(out_dir, b) for b in bins}
    centroids = {b: dfs[b].groupby(level="pos").median() for b in bins}

    presence = {}
    for b in bins:
        for (lemma, pos) in dfs[b].index:
            if pos in content_pos:
                presence.setdefault((lemma, pos), []).append(b)

    rows = []
    for (lemma, pos), labs in presence.items():
        if len(labs) < min_bins:
            continue
        first, last = labs[0], labs[-1]
        ve = dfs[first].loc[(lemma, pos)]
        vl = dfs[last].loc[(lemma, pos)]
        feats = ve.index.union(vl.index)
        a = ve.reindex(feats, fill_value=0.0).values
        c = vl.reindex(feats, fill_value=0.0).values
        drift = 0.5 * np.abs(a - c).sum()

        toward, best = None, -np.inf
        for cat in centroids[last].index:
            if cat == pos or cat not in centroids[first].index:
                continue
            gain = _cos(vl, centroids[last].loc[cat]) - _cos(ve, centroids[first].loc[cat])
            if gain > best:
                best, toward = gain, cat

        rows.append({"lemma": lemma, "pos": pos, "n_bins": len(labs),
                     "first": first, "last": last, "drift": round(drift, 4),
                     "toward": toward, "functional": toward in functional_pos,
                     "sim_gain": round(best, 4)})

    out = pd.DataFrame(rows).sort_values("drift", ascending=False)
    return out.head(top_n).reset_index(drop=True)


def feature_delta(lemma, pos, out_dir=OUT_DIR, top_k=25):
    """Top features by absolute change between first and last bin of presence."""
    bins = _bins_present(out_dir)
    present = []
    for b in bins:
        df = load_bin_profile_df(out_dir, b)
        if (lemma, pos) in df.index:
            v = df.loc[(lemma, pos)]
            present.append((b, v[v != 0.0].to_dict()))
    if len(present) < 2:
        print(f"{lemma}/{pos} present in <2 bins."); return None
    (_, e), (_, l) = present[0], present[-1]
    feats = set(e) | set(l)
    rows = [{"feature": f, "first": e.get(f, 0.0), "last": l.get(f, 0.0),
             "delta": l.get(f, 0.0) - e.get(f, 0.0)} for f in feats]
    df = pd.DataFrame(rows)
    df["abs_delta"] = df["delta"].abs()
    return df.sort_values("abs_delta", ascending=False).head(top_k).reset_index(drop=True)


# ===========================================================================
# PASS 2 plots
# ===========================================================================

def _int_ticks(ax, values):
    ax.set_xticks(sorted(set(values)))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_stability_lines(out_dir=OUT_DIR, save=None):
    bins = _bins_present(out_dir)
    diag = {b: pd.Series(np.diag(load_matrix(out_dir, b, "stability").values),
                         index=load_matrix(out_dir, b, "stability").index) for b in bins}
    df = pd.DataFrame(diag).T
    fig, ax = plt.subplots(figsize=(10, 6))
    for cat in df.columns:
        ax.plot(df.index, df[cat], marker="o", ms=4, label=cat)
    _int_ticks(ax, bins)
    ax.set_xlabel("Bin"); ax.set_ylabel("Stability (internal cohesion)")
    ax.set_title("Category stability over time"); ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    if save: fig.savefig(save, dpi=150)
    plt.show(); plt.close()
    return df


def plot_intruder_cells(cells=INTRUDER_CELLS, out_dir=OUT_DIR, save=None):
    bins = _bins_present(out_dir)
    series = {f"{o}->{r}": [] for o, r in cells}
    for b in bins:
        m = load_matrix(out_dir, b, "intruder")
        for o, r in cells:
            series[f"{o}->{r}"].append(m.loc[o, r] if (o in m.index and r in m.columns) else np.nan)
    fig, ax = plt.subplots(figsize=(10, 6))
    for k, vals in series.items():
        ax.plot(bins, vals, marker="o", ms=4, label=k)
    _int_ticks(ax, bins)
    ax.set_xlabel("Bin"); ax.set_ylabel("Intruder weight")
    ax.set_title("Selected categorical overlaps over time"); ax.legend()
    fig.tight_layout()
    if save: fig.savefig(save, dpi=150)
    plt.show(); plt.close()
    return pd.DataFrame(series, index=bins)


def plot_neighbourhood_lines(lemma, pos, allu=None, pcols=None,
                             out_dir=OUT_DIR, min_share=0.05, save=None):
    """One line per category: how much of the word's neighbourhood it occupies."""
    if allu is None:
        allu, pcols = load_all_units(out_dir)
    sub = allu[(allu.lemma == lemma) & (allu.pos == pos)].sort_values("bin")
    if sub.empty:
        print(f"{lemma}/{pos} absent."); return None
    keep = [c for c in pcols if sub[c].max() > min_share]
    fig, ax = plt.subplots(figsize=(10, 6))
    for c in keep:
        ax.plot(sub["bin"], sub[c], marker="o", ms=4, label=c[2:])
    _int_ticks(ax, sub["bin"])
    ax.set_xlabel("Bin"); ax.set_ylabel("Neighbourhood share (p_norm)")
    ax.set_title(f"Neighbourhood composition: {lemma} / {pos}")
    ax.legend(ncol=4, fontsize=8); fig.tight_layout()
    if save: fig.savefig(save, dpi=150)
    plt.show(); plt.close()
    return sub.set_index("bin")[keep]


def plot_trajectory(lemma, pos, allu=None, out_dir=OUT_DIR, save=None):
    if allu is None:
        allu, _ = load_all_units(out_dir)
    sub = allu[(allu.lemma == lemma) & (allu.pos == pos)].sort_values("bin")
    if sub.empty:
        print(f"{lemma}/{pos} absent."); return None
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(sub.purity, sub.entropy, c=sub["bin"], cmap="viridis", s=90, zorder=3)
    for _, row in sub.iterrows():
        ax.annotate(str(int(row["bin"])), (row.purity, row.entropy),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    fig.colorbar(sc, label="Bin")
    ax.set_xlabel("Purity"); ax.set_ylabel("Entropy")
    ax.set_xlim(-0.05, 1.05)
    ax.set_title(f"State-space: {lemma} / {pos}\n(low purity + low entropy = focused bridge)")
    fig.tight_layout()
    if save: fig.savefig(save, dpi=150)
    plt.show(); plt.close()
    return sub[["bin", "purity", "entropy", "rival"]]


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    source_decade_crosstab()                       # check the confound first

    bins = prepare_bin_dirs(build_bin_groups())
    run_pass1(bins)

    allu, pcols = load_all_units()
    plot_stability_lines(save=os.path.join(OUT_DIR, "stability_lines.png"))
    plot_intruder_cells(save=os.path.join(OUT_DIR, "intruder_cells.png"))

    movers = discover_movers()
    movers.to_csv(os.path.join(OUT_DIR, "discovered_movers.csv"), index=False)
    print("\nFunctionalisation candidates (content word -> functional category):")
    print(movers[movers.functional].head(20).to_string(index=False))
    print("\nAll top movers:")
    print(movers.head(20).to_string(index=False))

    for lemma, pos in WATCHLIST:
        plot_neighbourhood_lines(lemma, pos, allu, pcols,
                                 save=os.path.join(OUT_DIR, f"nbh_{lemma}_{pos}.png"))
        plot_trajectory(lemma, pos, allu,
                        save=os.path.join(OUT_DIR, f"traj_{lemma}_{pos}.png"))
        fd = feature_delta(lemma, pos)
        if fd is not None:
            print(f"\nFeature delta {lemma}/{pos}:\n", fd.to_string(index=False))