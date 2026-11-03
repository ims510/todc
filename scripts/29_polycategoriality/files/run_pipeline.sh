#!/usr/bin/env bash
# Run the full pipeline. Edit the paths at the top to point at YOUR treebank.
#
# Stage outputs are written under outputs/ (analysis CSVs/TXTs, training logs,
# evaluation reports). Models go under models/. Predictions go under preds/.
#
# Usage:  bash run_pipeline.sh
set -euo pipefail

# ----------------- EDIT THESE ----------------------------------------------
TRAIN=${TRAIN:-/Users/madalina/Documents/PHD/code/data/small_test/UD_French-GSD/fr_gsd-ud-train.conllu}
DEV=${DEV:-/Users/madalina/Documents/PHD/code/data/small_test/UD_French-GSD/fr_gsd-ud-dev.conllu}
TEST=${TEST:-/Users/madalina/Documents/PHD/code/data/small_test/UD_French-GSD/fr_gsd-ud-test.conllu}
EPOCHS=${EPOCHS:-10}
TOPK=${TOPK:-30}
RULES=${RULES:-obl=nmod}    # comma-separated, e.g. "obl=nmod,csubj=nsubj"
# ---------------------------------------------------------------------------

mkdir -p outputs models preds

echo "[1] Building relation x (head_upos, dep_upos) table"
python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/collapse_relationships/01_build_table.py' "$TRAIN" outputs

echo "[2] Computing TVD between every relation pair"
python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/collapse_relationships/02_compute_tvd.py' outputs/01_relation_pos_table.csv outputs

echo "[3] Computing per-pair classification accuracy and conditional entropy"
python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/collapse_relationships/03_pair_analysis.py' "$TRAIN" outputs/02_tvd_ranking.csv outputs "$TOPK"

echo "[4] Collapsing labels (rules: $RULES)"
# 04_collapse.py accepts --rules to collapse labels using the same rule set
# that is later used for uncollapsing and evaluation.
python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/collapse_relationships/04_collapse.py' "$TRAIN" outputs/train_collapsed.conllu --rules "$RULES"
python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/collapse_relationships/04_collapse.py' "$DEV"   outputs/dev_collapsed.conllu --rules "$RULES"
python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/collapse_relationships/04_collapse.py' "$TEST"  outputs/test_collapsed.conllu --rules "$RULES"

echo "[5a] Training parser on ORIGINAL labels"
python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/collapse_relationships/05_train_parser.py' train "$TRAIN" "$DEV" models/orig --epochs "$EPOCHS"

echo "[5b] Training parser on COLLAPSED labels"
python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/collapse_relationships/05_train_parser.py' train outputs/train_collapsed.conllu \
                                outputs/dev_collapsed.conllu \
                                models/collapsed --epochs "$EPOCHS"

echo "[6a] Predicting + scoring ORIGINAL parser"
python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/collapse_relationships/05_train_parser.py' predict models/orig "$TEST" preds/test_orig.conllu
python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/collapse_relationships/06_evaluate.py' compare "$TEST" preds/test_orig.conllu \
       --out outputs/06_eval_original.txt

echo "[6b] Predicting + scoring COLLAPSED parser (against collapsed gold)"
python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/collapse_relationships/05_train_parser.py' predict models/collapsed outputs/test_collapsed.conllu \
                                                    preds/test_collapsed.conllu
python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/collapse_relationships/06_evaluate.py' compare outputs/test_collapsed.conllu preds/test_collapsed.conllu \
       --out outputs/06_eval_collapsed.txt

echo "[6c] Uncollapsing predictions and scoring against ORIGINAL gold"
python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/collapse_relationships/06_evaluate.py' uncollapse "$TRAIN" "$TEST" \
                                 preds/test_collapsed.conllu \
                                 preds/test_uncollapsed.conllu \
                                 --rules "$RULES" \
                                 --out outputs/06_eval_uncollapsed.txt

echo
echo "Done. Inspect outputs/ for CSVs and reports."
