import grewpy
from grewpy import Corpus, CorpusDraft, Request
from collections import Counter
import sys
sys.path.insert(1, "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/tod")

import tod.corpus
import tod.outliers
import tod.clustering
import tod.plotting
import tod.dimension_reduction_classic

treebank_path = "/Users/madalina/Downloads/bUD_English-GUM"
grew_pattern = "pattern{X[upos=AUX|VERB]}"
patterns_text_file = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/3. probability_matrix/patterns_verbsaux.txt"

corpus = tod.corpus.Corpus(
    treebank_path=treebank_path,
    grew_pattern=grew_pattern,
    patterns_text_file=patterns_text_file,
    matrix_type="coverage",
    excluded_feature_patterns=[r"CxnElt=", r"Cxn=", r"XML=", r"PDTB=", r"SplitAnte=", r"MSeg=", r"Entity=", r"Discourse=", r"Bridge=", r"own"]
)


clustering = tod.clustering.SparseKMeans(corpus=corpus, k=2, top_n_features=20)
dim_red = tod.dimension_reduction_classic.Tsne_corpus(corpus, n_components=2)
fig = tod.plotting.cluster_scatter_plot(corpus, dim_red, clustering)
fig.write_html("sparse_kmeams_english_gum_verbsaux_k=2_noown.html")