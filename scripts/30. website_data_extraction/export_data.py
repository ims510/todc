"""
export_data.py
==============
Export syntactic-only treebanks (your 60-treebank UD loop).

Output layout
-------------
  site/data/
    index.json                          ← updated with each treebank's entry
    <Treebank-Name>/
      syntactic/
        stability.json
        intruder.json
        bridge_pairs.json

For prosody-capable treebanks (Rhapsodie), use export_prosody.py — it writes
additional space subdirectories alongside syntactic/.

Usage
-----
  python export_data.py
"""

import os
import sys
import traceback

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Bring tod.corpus onto the path
sys.path.insert(1, "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/tod")
import tod.corpus

import tidy_analysis as ta

# =============================================================================
# CONFIG — edit before running
# =============================================================================
TREEBANKS_FOLDER   = "/Users/madalina/Documents/PHD/code/data/test"
PATTERNS_TEXT_FILE = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/3. probability_matrix/patterns_all_nodes.txt"
GREW_PATTERN       = "pattern{X[upos<>PUNCT]}"
OUTPUT_DIR         = "./site/data"

# If you also run export_prosody.py for Rhapsodie, skip it here so this loop
# doesn't write a stripped-down syntactic-only version of it.
SKIP_TREEBANKS = {"UD_French-Rhapsodie", "SUD_French-Rhapsodie"}

# Same feature filters as before
EXCLUDED_FEATURE_PATTERNS = [r"own"]
INCLUDED_FEATURE_PATTERNS = [
    r"upos=",      r"position=",  r"rel_shallow=",
    r"Aspect=",    r"Animacy=",   r"Case=",
    r"Clusivity=", r"Definite=",  r"Deixis=",     r"DeixisRef",
    r"Evident=",   r"Negation=",  r"Number=",     r"Gender=",
    r"Degree=",    r"ExtPos=",    r"Foreign=",    r"Mood",
    r"NounClass=", r"NumType=",   r"Person=",     r"Polarity=",
    r"Polite=",    r"Poss=",      r"PronType=",   r"Reflex=",
    r"Tense=",     r"Typo=",      r"VerbForm=",   r"Voice=",
]


# =============================================================================
# Per-treebank export
# =============================================================================

def export_treebank(treebank_path, treebank_name):
    corpus = tod.corpus.Corpus(
        treebank_path             = treebank_path,
        grew_pattern              = GREW_PATTERN,
        patterns_text_file        = PATTERNS_TEXT_FILE,
        use_sud                   = True,
        matrix_type               = "coverage",
        excluded_feature_patterns = EXCLUDED_FEATURE_PATTERNS,
        included_feature_patterns = INCLUDED_FEATURE_PATTERNS,
    )
    similarity_matrix = cosine_similarity(corpus.feature_matrix)

    output_dir = os.path.join(OUTPUT_DIR, treebank_name, "syntactic")
    was, n_cats = ta.export_space(
        output_dir          = output_dir,
        idx2lexunit         = corpus._idx2lexunit,
        idx2feature         = corpus._idx2feature,
        lexunit2idx         = corpus._lexunit2idx,
        feature_matrix      = corpus.feature_matrix,
        similarity_matrix   = similarity_matrix,
    )

    ta.update_index(
        index_path     = os.path.join(OUTPUT_DIR, "index.json"),
        treebank_name  = treebank_name,
        spaces_info    = {"syntactic": {"was": was, "n_categories": n_cats}},
        default_space  = "syntactic",
    )
    return was, n_cats


# =============================================================================
# Main loop
# =============================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    treebanks = sorted([
        t for t in os.listdir(TREEBANKS_FOLDER)
        if t.startswith("UD_") and t not in SKIP_TREEBANKS
    ])
    print(f"Found {len(treebanks)} treebanks in {TREEBANKS_FOLDER}\n")

    for treebank_dir in tqdm(treebanks, desc="Exporting"):
        full_path     = os.path.join(TREEBANKS_FOLDER, treebank_dir)
        treebank_name = "-".join(treebank_dir.split("_")[1:])

        try:
            was, n_cats = export_treebank(full_path, treebank_name)
            print(f"  {treebank_name}: WAS={was:.3f}  ({n_cats} categories)")
        except Exception as e:
            print(f"  {treebank_name}: FAILED — {e}")
            traceback.print_exc()

    print(f"\nDone. Index at {OUTPUT_DIR}/index.json")
    print("Now run: python export_prosody.py  (if you want Rhapsodie too)")