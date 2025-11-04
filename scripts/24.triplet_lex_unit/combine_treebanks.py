"""
This script combines multiple treebanks into a single treebank, adding the name of the original treebank in the metadata of each sentence.
The output is a single conllu file, that can then be used for clustering when the lexical units are represented by 3 elements (lemma, upos, treebank).
"""
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


treebank_paths = ["/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/data/input/Universal_Dependencies/ud-treebanks-v2.15/UD_French-Sequoia", 
                  "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/data/input/Universal_Dependencies/ud-treebanks-v2.15/UD_French-GSD"]

# treebank_paths = ["/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/24.triplet_lex_unit/test1.conllu", "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/24.triplet_lex_unit/test2.conllu"]
output_path = "combined_tb.conllu"
with open(output_path, "w", encoding="utf-8") as output_file:
    output_file.write(
                        "# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC \n"
                    )
    for tb_path in treebank_paths:
        grewpy.set_config("ud")  
        corpus = Corpus(tb_path)  
        draft = CorpusDraft(corpus)

        tb_name = tb_path.split("/")[-1].replace(".conllu", "")
        for sent_id in draft:
            conll_string = draft[sent_id].to_conll()
            # output_file.write(f"# sent_id = {sent_id} \n")
            output_file.write(f"# treebank = {tb_name} \n")
            output_file.write(conll_string)
            output_file.write("\n")


treebank_path = output_path
grew_pattern = "pattern{X[upos<>PUNCT]}"
patterns_text_file = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/3. probability_matrix/patterns_all_nodes.txt"
analysed_category = "all_nodes"

corpus = tod.corpus.CorpusTriplet(
    treebank_path=treebank_path,
    grew_pattern=grew_pattern,
    patterns_text_file=patterns_text_file,
    matrix_type="coverage",
    min_occurrences=10
)

corpus.find_far_apart_lexunits(metric="cosine", top_k=100, print_results=True)

# pair_counts = Counter(t[:2] for t in lex_units)
# if pair_counts[t[:2]]

# clustering = tod.clustering.SparseKMeans(corpus=corpus, k=10, top_n_features=10)
# dim_red = tod.dimension_reduction_classic.Tsne_corpus(corpus, n_components=2)
# fig = tod.plotting.cluster_scatter_plot_shapes(corpus, dim_red, clustering)
# fig.write_html("test.html")