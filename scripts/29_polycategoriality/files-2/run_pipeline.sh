#!/usr/bin/env bash
# Three-phase pipeline:
#
#   1. analysis : stages 1-3. Statistical TVD/entropy screen produces
#                 outputs/03_suggested_rules.txt (cheap, overgenerates).
#
#   2. validate : stage 3b. For each candidate, applies the collapse as a
#                 column-merge on the feature matrix from your tod corpus,
#                 recomputes the stability matrix, and keeps only the rules
#                 that don't damage POS purity (ΔWAS, worst diagonal,
#                 max offdiag). Produces outputs/03b_validated_rules.txt.
#
#   3. train    : stages 4-6. Collapses the conllu files using the
#                 validated rules, trains parsers, evaluates with
#                 deterministic uncollapse via head_upos + dep_upos.
#
# Usage:
#   bash run_pipeline.sh analysis
#   bash run_pipeline.sh validate
#   bash run_pipeline.sh train
#   bash run_pipeline.sh all
#
# Or skip stage 3b and use inline rules:
#   RULES="obl=nmod,csubj=nsubj" bash run_pipeline.sh train
set -euo pipefail

# ----------------- EDIT THESE ----------------------------------------------
TRAIN=${TRAIN:-/Users/madalina/Documents/PHD/code/data/small_test/UD_French-GSD/fr_gsd-ud-train.conllu}
DEV=${DEV:-/Users/madalina/Documents/PHD/code/data/small_test/UD_French-GSD/fr_gsd-ud-dev.conllu}
TEST=${TEST:-/Users/madalina/Documents/PHD/code/data/small_test/UD_French-GSD/fr_gsd-ud-test.conllu}

# Stage 3b needs the tod corpus inputs:
TREEBANK_DIR=${TREEBANK_DIR:-/Users/madalina/Documents/PHD/code/data/small_test/UD_French-GSD/}
PATTERNS_FILE=${PATTERNS_FILE:-/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/3. probability_matrix/patterns_all_nodes.txt}
export TOD_PATH=${TOD_PATH:-/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/tod/tod}

# Thresholds for stage 3b (looser = more rules kept)
WAS_THRESHOLD=${WAS_THRESHOLD:-0.005}
DIAG_THRESHOLD=${DIAG_THRESHOLD:-0.02}
OFFDIAG_THRESHOLD=${OFFDIAG_THRESHOLD:-0.02}

EPOCHS=${EPOCHS:-10}
TOPK=${TOPK:-500}
# ---------------------------------------------------------------------------

CMD=${1:-all}

run_analysis() {
  mkdir -p outputs
  echo "[1] Building relation x (head_upos, dep_upos) table (skipping punct/root/PUNCT)"
  python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/files-2/01_build_table.py' "$TRAIN" outputs

  echo "[2] Computing TVD between every relation pair"
  python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/files-2/02_compute_tvd.py' outputs/01_relation_pos_table.csv outputs

  echo "[3] Per-pair classification accuracy and conditional entropy"
  python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/files-2/03_pair_analysis.py' "$TRAIN" outputs/02_tvd_ranking.csv outputs "$TOPK"

  echo
  echo "==> Stage 3 candidates in outputs/03_suggested_rules.txt"
  echo "==> Next: bash run_pipeline.sh validate"
}

run_validate() {
  mkdir -p outputs
  echo "[3b] Validating candidates with the category-purity test"
  python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/files-2/03b_validate_candidates.py' \
      --treebank "$TREEBANK_DIR" \
      --patterns-file "$PATTERNS_FILE" \
      --candidates outputs/03_suggested_rules.txt \
      --output-dir outputs \
      --was-threshold "$WAS_THRESHOLD" \
      --diag-threshold "$DIAG_THRESHOLD" \
      --offdiag-threshold "$OFFDIAG_THRESHOLD" \
      --also-test-combined

  echo
  echo "==> Validated rules in outputs/03b_validated_rules.txt"
  echo "==> Full report in outputs/03b_validation_report.txt"
  echo "==> Next: bash run_pipeline.sh train"
}

run_train_eval() {
  mkdir -p outputs models preds

  if [[ -n "${RULES:-}" ]]; then
    echo "Using inline rules: $RULES"
    RULES_ARG=(--rules "$RULES")
  else
    RULES_FILE=outputs/03b_validated_rules.txt
    if [[ ! -f "$RULES_FILE" ]]; then
      echo "ERROR: $RULES_FILE not found. Run 'bash run_pipeline.sh validate' first,"
      echo "       or set RULES env var: RULES=\"obl=nmod\" bash run_pipeline.sh train"
      exit 1
    fi
    echo "Using validated rules from: $RULES_FILE"
    RULES_ARG=(--rules-file "$RULES_FILE")
  fi

  echo "[4] Collapsing labels in train/dev/test"
  python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/files-2/04_collapse.py' "$TRAIN" outputs/train_collapsed.conllu "${RULES_ARG[@]}"
  python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/files-2/04_collapse.py' "$DEV"   outputs/dev_collapsed.conllu   "${RULES_ARG[@]}"
  python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/files-2/04_collapse.py' "$TEST"  outputs/test_collapsed.conllu  "${RULES_ARG[@]}"

  echo "[5a] Training parser on ORIGINAL labels"
  python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/files-2/05_train_parser.py' train "$TRAIN" "$DEV" models/orig --epochs "$EPOCHS"

  echo "[5b] Training parser on COLLAPSED labels"
  python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/files-2/05_train_parser.py' train outputs/train_collapsed.conllu \
                                  outputs/dev_collapsed.conllu \
                                  models/collapsed --epochs "$EPOCHS"

  echo "[6a] Predicting + scoring ORIGINAL parser"
  python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/files-2/05_train_parser.py' predict models/orig "$TEST" preds/test_orig.conllu
  python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/files-2/06_evaluate.py' compare "$TEST" preds/test_orig.conllu \
         --out outputs/06_eval_original.txt

  echo "[6b] Predicting + scoring COLLAPSED parser (vs collapsed gold)"
  python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/files-2/05_train_parser.py' predict models/collapsed outputs/test_collapsed.conllu \
                                                      preds/test_collapsed.conllu
  python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/files-2/06_evaluate.py' compare outputs/test_collapsed.conllu preds/test_collapsed.conllu \
         --out outputs/06_eval_collapsed.txt

  echo "[6c] Uncollapsing predictions, scoring against ORIGINAL gold"
  python3.11 '/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/29. polycategoriality/files-2/06_evaluate.py' uncollapse "$TRAIN" "$TEST" \
                                   preds/test_collapsed.conllu \
                                   preds/test_uncollapsed.conllu \
                                   "${RULES_ARG[@]}" \
                                   --out outputs/06_eval_uncollapsed.txt

  echo
  echo "Done. See outputs/06_eval_{original,collapsed,uncollapsed}.txt for the three LAS numbers."
}

case "$CMD" in
  analysis)   run_analysis ;;
  validate)   run_validate ;;
  train)      run_train_eval ;;
  all)        run_analysis; run_validate; run_train_eval ;;
  *) echo "Usage: $0 {analysis|validate|train|all}"; exit 1 ;;
esac
