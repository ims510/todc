import csv
import grewpy
from grewpy import Corpus, CorpusDraft, Request


def get_dict_from_csv(file_path):
    result = {}
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            lex_unit, cluster_number = row
            lex_unit_elements = lex_unit.split(", ")
            lex_unit_tuple = []
            for elem in lex_unit_elements:
                lex_unit_tuple.append(elem.strip("()'"))
            lex_unit = tuple(lex_unit_tuple)
            result[lex_unit] = int(cluster_number)
    return result

def add_cluster_notation(graph):
    for node in graph:
        if 'lemma' in graph[node]:
            lex_unit = (graph[node]['lemma'], graph[node]['upos'])
            if lex_unit in cluster_assignments:
                cluster_id = cluster_assignments[lex_unit]
                graph[node]["Cluster"] = str(cluster_id)
    return graph

def annotate_treebank_with_clusters(treebank_path, cluster_assignments, config, output_path):
    grewpy.set_config(config)  
    corpus = Corpus(treebank_path)  
    draft = CorpusDraft(corpus)

    output_draft = draft.map(add_cluster_notation)
    conll_string= output_draft.to_conll()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC\n")
        f.write(conll_string)

treebank_path = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/data/input/Universal_Dependencies/ud-treebanks-v2.15/UD_French-GSD"
output_path = "all_nodes_cluster_annotated.conllu"
cluster_assignment_csv = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/20.sparse_vs_grex/all_nodes_cluster_assignments.csv"
config = "ud"

cluster_assignments = get_dict_from_csv(cluster_assignment_csv)
annotate_treebank_with_clusters(treebank_path, cluster_assignments, config, output_path)


