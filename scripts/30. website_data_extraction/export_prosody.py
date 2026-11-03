"""
export_prosody.py
=================
Export prosody-capable treebanks (SUD-French-Rhapsodie, pSUD-Naija-NSC, …)
with several feature spaces each:
  syntactic, prosodic, prosodic_foot, prosodic_group, prosodic_package,
  prosodic_period, prosodic_all_levels, combined.

Each space gets its own subdirectory with the same JSON schema as
export_data.py (now including features.json), so the website handles all
treebanks uniformly.

To add another prosodic treebank, just append an entry to PROSODY_TREEBANKS.

Usage
-----
  python export_prosody.py
"""

import os
import sys
import yaml

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# CONFIG
# =============================================================================
TOD_PATH      = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/tod"
GREX_PATH     = "/Users/madalina/Documents/M2TAL/stage/grex/grex2"
PATTERNS_FILE = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/3. probability_matrix/patterns_all_nodes_prosody.txt"

OUTPUT_DIR    = "./site/data"
DEFAULT_SPACE = "syntactic"     # which space's WAS goes on the landing chart
GREW_PATTERN  = "pattern{X[upos<>PUNCT]}"
MIN_LEXUNITS  = 10
N_QUANTILE_BINS = 3

# Add a dict here for every prosodic treebank you want on the site.
PROSODY_TREEBANKS = [
    {
        "name": "French-Rhapsodie",
        "path": "/Users/madalina/Documents/PHD/code/data/SUD_French-Rhapsodie/prosody_pauses",
    },
    {
        "name": "Naija-NSC",
        "path": "/Users/madalina/Documents/PHD/code/data/pSUD_Naija-NSC",
    },
]


# =============================================================================
# Third-party setup
# =============================================================================
for p in (TOD_PATH, GREX_PATH):
    if p not in sys.path:
        sys.path.insert(1, p)

import pyximport
pyximport.install()

import grewpy
import grex.utils
import grex.data_prosody
grewpy.set_config("sud")

import tidy_analysis as ta


# =============================================================================
# Feature handling (lifted from the notebook, with Naija additions)
# =============================================================================

FEATURES_TO_DROP = {
    'MeanF0', 'MeanF0Step', 'AvgAmplitude', 'MaxAmplitude',
    'SylForm', 'AlignBegin', 'AlignEnd', 'pauseID',
    'BeforeMeanF0', 'BeforeMeanF0Step', 'BeforeAvgAmplitude',
    'NextMeanF0',   'NextMeanF0Step',   'NextAvgAmplitude',
    'Semiton', 'Overlap', 'Loc', 'Glo', 'Backchannel',
    # Naija additions
    'AvgAmplitudeNormalized', 'Nucleus', 'Onset',
    'MeanF0GlobalZscore', 'MeanF0LocalZscore', 'MeanF0Normalized',
}

FEATURES_TO_BIN = {
    'SemitonesFromUtteranceMean', 'Duration',
    'BeforeSemitonesFromUtterance', 'NextSemitonesFromUtterance',
    'BeforeDuration', 'NextDuration',
    'BeforeSemiton', 'NextSemiton', 'NextSemitonsFromUtterance',
    # Naija additions
    'AvgAmplitudeGlobalZscore', 'DurationGlobalZscore',
}

BIN_LABELS = {
    'SemitonesFromUtteranceMean':   ['F0_low',       'F0_mid',       'F0_high'],
    'Duration':                     ['Dur_short',    'Dur_mid',      'Dur_long'],
    'BeforeSemitonesFromUtterance': ['BeforeF0_low', 'BeforeF0_mid', 'BeforeF0_high'],
    'NextSemitonesFromUtterance':   ['NextF0_low',   'NextF0_mid',   'NextF0_high'],
}

SLAM_TONE_FEATURES = {'FootTone', 'GroupTone', 'PackageTone', 'Period_tone'}
_PITCH_LEVELS = {'L': 1, 'l': 2, 'm': 3, 'h': 4, 'H': 5}
_TIERS        = {'1', '2', '3'}

EXCLUDED_FEATURE_PATTERNS = [
    'own', 'ExtPos', 'Loc=', 'direction=flat', 'has_peak=False',
    'FootTone:pause', 'GroupTone:pause', 'PackageTone:pause',
    'Period_tone:pause', 'ProminenceFinal=Pause', 'ProminenceInitial=Pause',
    # Naija additions
    'AvgAmplitudeLocalZscore', 'DurationLocalZscore', 'MaxAmplitudeLocalZscore',
    'DurationNormalized', 'MaxAmplitudeGlobalZscore', 'Nucleus', 'Onset',
    'MeanF0GlobalZscore', 'MeanF0LocalZscore', 'MeanF0Normalized',
    'MaxAmplitudeNormalized', 'Coda',
]


def _parse_slam(label):
    if label == 'Pause':
        return {'pause': True, 'start': None, 'end': None, 'peak': None, 'tier': None}
    if (len(label) == 2
            and label[0] in _PITCH_LEVELS and label[1] in _PITCH_LEVELS):
        return {'pause': False, 'start': label[0], 'end': label[1],
                'peak': None, 'tier': None}
    if (len(label) == 4
            and label[0] in _PITCH_LEVELS and label[1] in _PITCH_LEVELS
            and label[2] in _PITCH_LEVELS and label[3] in _TIERS):
        return {'pause': False, 'start': label[0], 'end': label[1],
                'peak': label[2], 'tier': label[3]}
    return None


def _slam_subfeatures(prefix, label):
    parsed = _parse_slam(label)
    if parsed is None:
        return
    if parsed['pause']:
        yield f"{prefix}:pause=True"
        return
    s, e = parsed['start'], parsed['end']
    yield f"{prefix}:start={s}"
    yield f"{prefix}:end={e}"
    delta = _PITCH_LEVELS[e] - _PITCH_LEVELS[s]
    direction = 'rise' if delta > 0 else ('fall' if delta < 0 else 'flat')
    yield f"{prefix}:direction={direction}"
    yield f"{prefix}:magnitude={abs(delta)}"
    if parsed['peak'] is not None:
        yield f"{prefix}:has_peak=True"
        yield f"{prefix}:peak={parsed['peak']}"
        yield f"{prefix}:tier={parsed['tier']}"
    else:
        yield f"{prefix}:has_peak=False"


def _normalise_duration(value_str):
    try:
        v = float(value_str)
    except (ValueError, TypeError):
        return None
    if v == 0.0:
        return None
    return v * 1000 if v < 10 else v


def _is_sentinel(value_str):
    try:
        return float(value_str) == 0.0
    except (ValueError, TypeError):
        return False


def _feature_name(key):
    return key[-1]


def collect_continuous_values(all_data_items):
    collector = {f: [] for f in FEATURES_TO_BIN}
    for _, _, features in all_data_items:
        for key, value in features.items():
            fname = _feature_name(key)
            if fname not in FEATURES_TO_BIN:
                continue
            values = value if isinstance(value, set) else {value}
            for v in values:
                if fname == 'Duration':
                    norm = _normalise_duration(v)
                    if norm is not None:
                        collector['Duration'].append(norm)
                elif not _is_sentinel(v):
                    try:
                        collector[fname].append(float(v))
                    except (ValueError, TypeError):
                        pass
    return collector


def compute_thresholds(collector, n_bins=N_QUANTILE_BINS):
    thresholds = {}
    for fname, values in collector.items():
        if not values:
            continue
        arr = np.array(values)
        quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
        thresholds[fname] = np.quantile(arr, quantiles).tolist()
    return thresholds


def _bin_value(fname, value_str, thresholds):
    if fname not in thresholds or _is_sentinel(value_str):
        return None
    if fname == 'Duration':
        v = _normalise_duration(value_str)
    else:
        try:
            v = float(value_str)
        except (ValueError, TypeError):
            return None
    if v is None:
        return None
    bounds = thresholds[fname]
    labels = BIN_LABELS.get(fname, [f'{fname}_Q1', f'{fname}_Q2', f'{fname}_Q3'])
    for i, b in enumerate(bounds):
        if v <= b:
            return labels[i]
    return labels[-1]


def format_features(features, thresholds):
    result = []
    for key, value in features.items():
        fname  = _feature_name(key)
        prefix = ':'.join(key)
        if fname in FEATURES_TO_DROP:
            continue
        if fname in FEATURES_TO_BIN:
            values = value if isinstance(value, set) else {value}
            for v in values:
                label = _bin_value(fname, v, thresholds)
                if label is not None:
                    result.append(f"{prefix}={label}")
            continue
        if fname in SLAM_TONE_FEATURES:
            values = value if isinstance(value, set) else {value}
            for v in values:
                result.extend(_slam_subfeatures(prefix, v))
            continue
        if isinstance(value, set):
            for v in value:
                result.append(f"{prefix}={v}")
        else:
            result.append(f"{prefix}={value}")
    return result


def _filter_features(features, excluded_patterns):
    import re
    return [f for f in features
            if not any(re.search(p, f) for p in excluded_patterns)]


# =============================================================================
# Corpus loading (now parameterised by path)
# =============================================================================

def load_prosody_treebank(treebank_path):
    """
    Returns: feature_matrix, idx2lexunit, lexunit2idx, idx2feature
    """
    print(f"  Loading corpus from {treebank_path} …")
    corpus = grewpy.Corpus(treebank_path)
    draft  = grewpy.CorpusDraft(corpus)

    all_matches = corpus.search(
        grewpy.Request(GREW_PATTERN)
            .without("X[InIdiom=Yes]")
            .without("X[Idiom=Yes]")
            .without("X[InTitle=Yes]")
            .without("X[Title=Yes]")
            .without("X[Scrap=Yes]")
            .without("X[Foreign]")
            .without("X[Lang]")
            .without("X-[fixed]->Y")
            .without("Y-[flat:name]->X")
            .without("Y-[goeswith]->X"),
        clustering_parameter=["X.lemma"],
    )
    matches = {k: v for k, v in all_matches.items() if len(v) > 10}

    sent_id_to_sentence = {
        draft[i].meta["sent_id"]: draft[i].features for i in range(len(draft))
    }

    match_upos = {}
    for lemma, occurrences in matches.items():
        for m in occurrences:
            sent_id = m["sent_id"]
            node_id = str(m["matching"]["nodes"]["X"])
            sent_feats = sent_id_to_sentence.get(sent_id)
            if sent_feats is None or node_id not in sent_feats:
                continue
            tok = sent_feats[node_id]
            pos = tok.get("ExtPos") or tok.get("upos")
            if pos is None:
                continue
            match_upos.setdefault((lemma, pos), []).append(m)
    match_upos = {k: v for k, v in match_upos.items() if len(v) > 10}
    print(f"    (lemma, POS) units after frequency filter: {len(match_upos)}")

    with open(PATTERNS_FILE) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    templates = grex.utils.FeaturePredicate.from_config(config["templates"])
    feature_predicate = grex.utils.FeaturePredicate.from_config(
        config["features"], templates=templates,
    )

    all_data_items = []
    for lex_unit, mts in match_upos.items():
        for match in mts:
            feats = grex.data_prosody.extract_features(draft, match, feature_predicate)
            all_data_items.append((lex_unit, match, feats))

    collector  = collect_continuous_values(all_data_items)
    thresholds = compute_thresholds(collector, n_bins=N_QUANTILE_BINS)

    data = {k: [] for k in match_upos}
    for lex_unit, _, features in all_data_items:
        formatted = _filter_features(
            format_features(features, thresholds),
            excluded_patterns=EXCLUDED_FEATURE_PATTERNS,
        )
        data[lex_unit].append(formatted)

    unique_lemma    = sorted(set(data))
    unique_features = sorted({feat for lists in data.values()
                                  for fs in lists for feat in fs})

    idx2feature = {i: feat for i, feat in enumerate(unique_features)}
    feature2idx = {feat: i for i, feat in idx2feature.items()}
    idx2lexunit = {i: lu  for i, lu  in enumerate(unique_lemma)}
    lexunit2idx = {lu:  i for i, lu  in idx2lexunit.items()}

    co = np.zeros((len(idx2lexunit), len(idx2feature)))
    for lex_unit, feature_lists in data.items():
        for feature_list in feature_lists:
            for feature in feature_list:
                co[lexunit2idx[lex_unit], feature2idx[feature]] += 1
    row_sums = co.sum(axis=1)
    coverage = np.divide(co, row_sums[:, None],
                         out=np.zeros_like(co), where=row_sums[:, None] != 0)

    col_nonzero = (co > 0).sum(axis=0)
    keep        = col_nonzero >= MIN_LEXUNITS
    n_before    = coverage.shape[1]
    coverage    = coverage[:, keep]

    kept_features = [idx2feature[i] for i in range(n_before) if keep[i]]
    idx2feature   = {i: f for i, f in enumerate(kept_features)}
    print(f"    Kept {int(keep.sum())} / {n_before} features (≥{MIN_LEXUNITS} units)")

    return coverage, idx2lexunit, lexunit2idx, idx2feature


# =============================================================================
# Feature-space masks
# =============================================================================

PROSODIC_MARKERS = (
    # syllable-level / continuous acoustic
    'childSyl', 'prevSyl', 'nextSyl',
    'Slope', 'Glo', 'Loc', 'Duration',
    'PitchRange', 'AvgHeight', 'Semiton',
    'F0', 'Fused',
    # token-level prosodic structural annotations
    'Foot', 'Group', 'Package', 'Period',
    'Prominence', 'Hesitation', 'RhythmGroup', 'NextBreakLength',
    'IU', 'Nucleus',
)

LEVEL_MARKERS = {
    'foot':    ('Foot',),
    'group':   ('Group',),
    'package': ('Package',),
    'period':  ('Period',),
}


def _mask_for(markers, idx2feature):
    return np.array([
        any(p in feat for p in markers)
        for feat in idx2feature.values()
    ])


def build_spaces(idx2feature):
    """Return {space_name: boolean_mask_over_columns}."""
    prosodic_mask   = _mask_for(PROSODIC_MARKERS, idx2feature)
    syntactic_mask  = ~prosodic_mask
    level_masks     = {lvl: _mask_for(m, idx2feature)
                       for lvl, m in LEVEL_MARKERS.items()}
    all_levels_mask = np.any(np.stack(list(level_masks.values())), axis=0)
    return {
        "syntactic":           syntactic_mask,
        "prosodic":            prosodic_mask,
        # "prosodic_foot":       level_masks['foot'],
        # "prosodic_group":      level_masks['group'],
        # "prosodic_package":    level_masks['package'],
        # "prosodic_period":     level_masks['period'],
        # "prosodic_all_levels": all_levels_mask,
        "combined":            np.ones_like(prosodic_mask),
    }


# =============================================================================
# Per-treebank driver
# =============================================================================

def export_one(treebank_name, treebank_path):
    print(f"\n=== {treebank_name} ===")
    feature_matrix, idx2lexunit, lexunit2idx, idx2feature = \
        load_prosody_treebank(treebank_path)
    spaces = build_spaces(idx2feature)

    spaces_info = {}
    for space_name, mask in spaces.items():
        if not mask.any():
            print(f"  {space_name:<22s} empty mask, skipped")
            continue
        sim = cosine_similarity(feature_matrix[:, mask])
        out_dir = os.path.join(OUTPUT_DIR, treebank_name, space_name)
        was, n_cats = ta.export_space(
            output_dir        = out_dir,
            idx2lexunit       = idx2lexunit,
            idx2feature       = idx2feature,
            lexunit2idx       = lexunit2idx,
            feature_matrix    = feature_matrix,
            similarity_matrix = sim,
            feature_mask      = mask,
        )
        spaces_info[space_name] = {"was": was, "n_categories": n_cats}
        print(f"  {space_name:<22s} WAS={was:.3f}  "
              f"({n_cats} categories, {int(mask.sum())} features)")

    default = DEFAULT_SPACE if DEFAULT_SPACE in spaces_info else next(iter(spaces_info))
    ta.update_index(
        index_path     = os.path.join(OUTPUT_DIR, "index.json"),
        treebank_name  = treebank_name,
        spaces_info    = spaces_info,
        default_space  = default,
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for tb in PROSODY_TREEBANKS:
        export_one(tb["name"], tb["path"])
    print(f"\nDone. {len(PROSODY_TREEBANKS)} prosodic treebanks written to "
          f"{OUTPUT_DIR}/index.json")