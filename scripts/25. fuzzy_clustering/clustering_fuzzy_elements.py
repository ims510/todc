import sys
sys.path.insert(1, "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/tod")

import tod.corpus
import tod.outliers
import tod.clustering
import tod.plotting
import tod.dimension_reduction_classic
import numpy as np

treebank_name = "UD_French-GSD"
corpus = tod.corpus.CorpusFromMatrix(
    matrix_path=f"/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/UD_French-GSD_features_gt2clusters.npy",
    lexunits_path=f"/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/UD_French-GSD_lexunits_gt2clusters.csv",
    feature_names_path=f"/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/UD_French-GSD_feature_names.npy",
    coocc_matrix_path=f"/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/UD_French-GSD_cooccurrence_gt2clusters.npy"
)



clustering = tod.clustering.SparseKMeans(corpus=corpus, k=6, top_n_features=10)
dim_red = tod.dimension_reduction_classic.Tsne_corpus(corpus, n_components=2)
fig = tod.plotting.cluster_scatter_plot(corpus, dim_red, clustering)
fig.write_html("sparse_kmeans_gsd_inbetween.html")