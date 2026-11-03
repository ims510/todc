# Treebank relation-collapse analysis

A small pipeline that asks: which dependency-relation distinctions in a UD
treebank are essentially determined by `(head_upos, dependent_upos)`? Then
collapses one such distinction, retrains a parser on the simplified label
set, and recovers the original labels deterministically.

## Why multiple files instead of one

Every stage produces an inspectable artefact (CSV, txt report, conllu file).
Splitting them lets you re-run stage 3 with a different `top_k`, or stage 4
with a different rule set, without redoing the parser. A single mega-script
would either need a lot of CLI flags or would force you to redo the
expensive parts whenever you tweak a parameter. The whole pipeline is
glued together by `run_pipeline.sh`.

## Files

```
utils.py              CoNLL-U parsing helpers shared by every stage
01_build_table.py     relation × (head_upos, dep_upos) count table
02_compute_tvd.py     total variation distance between every relation pair
03_pair_analysis.py   per-pair majority-class accuracy + conditional entropy
04_collapse.py        rewrite a CoNLL-U file with collapsed labels
05_train_parser.py    small biaffine PyTorch parser (train + predict)
06_evaluate.py        UAS/LAS scoring + deterministic uncollapse
run_pipeline.sh       end-to-end runner
```

## Quick start

```bash
# install deps
pip install pandas numpy torch

# edit paths in run_pipeline.sh, then:
bash run_pipeline.sh
```

Or run stages individually, with the defaults wired through `outputs/`.

## What each stage produces

Stage 1 — `outputs/01_relation_pos_table.csv` and `01_summary.txt`.
Rows are base relations (`nsubj:pass` → `nsubj`), columns are
`HEAD_UPOS->DEP_UPOS` strings, cells are token counts.

Stage 2 — `outputs/02_tvd_ranking.csv` and `02_tvd_top.txt`. Every
pair `(r1, r2)` and its TVD over the (head, dep) UPOS distribution.
TVD = 1 means disjoint supports.

Stage 3 — `outputs/03_pair_analysis.csv`,
`03_deterministic_pairs.txt`, and per-pair breakdowns under
`03_pair_breakdowns/`. The breakdowns show, for each cell, which
relation wins and by how much.

The "deterministic given POS" flag fires when

  - majority-class-per-cell accuracy > 0.95, **OR**
  - conditional entropy `H(R | head_upos, dep_upos) < 0.10`.

Stage 4 — collapsed CoNLL-U files. The default rule is `obl → nmod`
(extend `DEFAULT_COLLAPSE_RULES` in `04_collapse.py` if stage 3 turns
up other candidates).

Stage 5 — `models/orig/` and `models/collapsed/`. Each contains
`model.pt`, `vocabs.json`, and `training_log.txt` with per-epoch
dev UAS/LAS.

Stage 6 — three eval reports under `outputs/`:

  - `06_eval_original.txt` — parser trained on original labels, scored
    against original gold.
  - `06_eval_collapsed.txt` — parser trained on collapsed labels,
    scored against collapsed gold (this is the "easier task" baseline).
  - `06_eval_uncollapsed.txt` — collapsed predictions deterministically
    uncollapsed via head UPOS, scored against the **original** gold.
    The derived uncollapse cells are listed in the report.

## What you should expect to see

If `obl/nmod` is genuinely deterministic given head UPOS:

  - LAS_collapsed ≥ LAS_original (the parser has fewer labels to confuse).
  - LAS_uncollapsed ≈ LAS_original to within ~0.5 absolute (the
    deterministic rule recovers most of what was lost by collapsing).

The gap between LAS_uncollapsed and LAS_original tells you how much of
the obl/nmod distinction the parser was getting from `head_upos` anyway,
versus from word features.

## Parser choice (stage 5)

Implemented from scratch in PyTorch: a Dozat & Manning (2017) biaffine
arc-factored parser with word + POS embeddings, a single BiLSTM layer,
and biaffine MLP scorers for arcs and labels. ~2-3M parameters. Greedy
decoding (argmax head per dependent, no MST) — fine for UAS/LAS which
are per-token. CPU-trainable on small treebanks. The full motivation is
in the docstring at the top of `05_train_parser.py`.
