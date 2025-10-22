import re
import sys
sys.path.insert(1, "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/tod")

import tod.corpus
import tod.outliers
import tod.clustering
import tod.plotting
import tod.dimension_reduction_classic

treebank_path = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/data/input/Universal_Dependencies/ud-treebanks-v2.15/UD_French-Sequoia"
grew_pattern = "pattern{X[upos<>PUNCT]}"
patterns_text_file = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/3. probability_matrix/patterns_all_nodes.txt"
analysed_category = "all_nodes"

corpus = tod.corpus.Corpus(
    treebank_path=treebank_path,
    grew_pattern=grew_pattern,
    patterns_text_file=patterns_text_file,
    matrix_type="coverage",
    excluded_feature_patterns=[r"upos=", r"rel_shallow="]
)

for feature, idx in corpus._feature2idx.items():
    print(f"{idx}. {feature}")

dim_red = tod.dimension_reduction_classic.Tsne_corpus(corpus, n_components=2)
clustering = tod.clustering.HierarchicalClustering(corpus)
fig = tod.plotting.cluster_scatter_plot(corpus, dim_red, clustering)
fig.write_html("hac_sequoia_nosynlabels_noupos.html")

# clustering = tod.clustering.SparseKMeans(corpus=corpus, k=10, top_n_features=10)
# dim_red = tod.dimension_reduction_classic.Tsne_corpus(corpus, n_components=2)
# fig = tod.plotting.cluster_scatter_plot(corpus, dim_red, clustering)
# fig.write_html("sparse_kmeans_sequoia_noupos.html")