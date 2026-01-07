import sys
sys.path.insert(1, "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/tod")

import tod.corpus
import tod.outliers
import tod.clustering
import tod.plotting
import tod.dimension_reduction_classic

# treebank_path = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/data/input/Universal_Dependencies/ud-treebanks-v2.15/UD_French-Sequoia"
# grew_pattern = "pattern{X[upos<>PUNCT]}"
# patterns_text_file = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/3. probability_matrix/patterns_all_nodes.txt"
# analysed_category = "all_nodes"


# treebank_path = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/data/input/sud-treebanks-v2.16/SUD_French-Sequoia"

treebank_path = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/data/input/Universal_Dependencies/ud-treebanks-v2.15/UD_Romanian-RRT"
grew_pattern = "pattern{X[upos<>PUNCT]}"
patterns_text_file = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/3. probability_matrix/patterns_all_nodes.txt"
analysed_category = "all_nodes"

corpus = tod.corpus.Corpus(
    treebank_path=treebank_path,
    grew_pattern=grew_pattern,
    patterns_text_file=patterns_text_file,
    use_sud=True,
    matrix_type="coverage",
    excluded_feature_patterns=[r"upos=NA", r"rel_shallow=", r"rel_deep="]
)

# for testing with no syntactic relations (just the dependencies, no labels), or no upos or no both
# corpus = tod.corpus.Corpus(
#     treebank_path=treebank_path,
#     grew_pattern=grew_pattern,
#     patterns_text_file=patterns_text_file,
#     matrix_type="coverage",
#     use_sud=True,
#     # excluded_feature_patterns=[r"rel_shallow="],
#     excluded_feature_patterns=[r"upos=ADP", r"upos=AUX", r"upos=CCONJ", r"upos=DET", r"upos=PRON", r"upos=SCONJ", r"upos=NUM", r"upos=INTJ", r"upos=ADV", r"upos=PROPN", r"upos=SYM" ],
#     # excluded_feature_patterns=[r"rel_shallow=", r"upos="]
# )
# corpus = tod.corpus.Corpus(
#     treebank_path=treebank_path,
#     grew_pattern=grew_pattern,
#     patterns_text_file=patterns_text_file,
#     matrix_type="coverage"
# )


clustering = tod.clustering.SparseKMeans(corpus=corpus, k=11, top_n_features=10)
dim_red = tod.dimension_reduction_classic.Tsne_corpus(corpus, n_components=2)
fig = tod.plotting.cluster_scatter_plot(corpus, dim_red, clustering)
fig.write_html("sparse_kmeans_romanian_rrt.html")