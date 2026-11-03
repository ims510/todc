import sys
sys.path.insert(1, "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/tod")

import tod.corpus
import tod.outliers
import tod.clustering
import tod.plotting
import tod.dimension_reduction_classic


treebank_path = "/Users/madalina/Downloads/corpora/ja_gsd-ud-merged.conllu"
grew_pattern = "pattern{X[upos<>PUNCT]}"
patterns_text_file = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/3. probability_matrix/patterns_all_nodes.txt"
analysed_category = "all_nodes"

corpus = tod.corpus.Corpus(
    treebank_path=treebank_path,
    grew_pattern=grew_pattern,
    patterns_text_file=patterns_text_file,
    use_sud=False,
    matrix_type="coverage",
    min_occurrences=3,
    excluded_feature_patterns=[r"UnidicInfo=", r"LUWPOS=", r"PrevUDLemma"]
)



clustering = tod.clustering.SparseKMeans(corpus=corpus, k=9, top_n_features=10, write_to_file=False)
dim_red = tod.dimension_reduction_classic.Tsne_corpus(corpus, n_components=2)
fig = tod.plotting.cluster_scatter_plot(corpus, dim_red, clustering)
fig.write_html("sparse_kmeans_ja.html")