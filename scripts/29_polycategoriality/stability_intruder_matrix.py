import grewpy
from grewpy import Corpus, CorpusDraft, Request
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.insert(1, "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/tod")

import tod.corpus

treebank_path = "/Users/madalina/Documents/PHD/code/data/test/UD_French-GSD"
# treebank_path = "/Users/madalina/Downloads/SUD_French-GSD"
# treebank_path = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/data/input/Universal_Dependencies/ud-treebanks-v2.15/UD_Chinese-GSD"
# treebank_path = "/Users/madalina/Documents/PHD/code/data/test/UD_Beja-Autogramm"
# treebank_path = "/Users/madalina/Downloads/bUD_English-GUM"
# treebank_path = "/Users/madalina/Documents/PHD/code/data/test/UD_Apurina-UFPA"
# treebank_path = "/Users/madalina/Documents/PHD/code/data/test/UD_Romanian-RRT"
grew_pattern = "pattern{X[upos<>PUNCT]}"
patterns_text_file = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/3. probability_matrix/patterns_all_nodes.txt"
analysed_category = "all_nodes"

corpus = tod.corpus.Corpus(
    treebank_path=treebank_path,
    grew_pattern=grew_pattern,
    patterns_text_file=patterns_text_file,
    use_sud=True,
    matrix_type="coverage",
    excluded_feature_patterns=[r"own"],
    included_feature_patterns=[r"upos=", 
                                r"position=", 
                                r"rel_shallow=",
                                r"Abbr=", 
                                r"Aspect=",  
                                r"Animacy=",  
                                r"Case=",  
                                r"Clusivity=",  
                                r"Definite=",  
                                r"Deixis=", 
                                r"DeixisRef",  
                                r"Evident=",  
                                r"Negation=",
                                r"Number=",  
                                r"Gender=",  
                                r"Degree=",  
                                r"ExtPos=", 
                                r"Foreign=", 
                                r"Mood",  
                                r"NounClass=",  
                                r"NumType=",  
                                r"Person=",  
                                r"Polarity=", 
                                r"Polite=",  
                                r"Poss=",  
                                r"PronType=",  
                                r"Reflex=",  
                                r"Tense=",  
                                r"Typo=",     
                                r"VerbForm=",  
                                r"Voice=",   ])
    
similarity_matrix = cosine_similarity(corpus.feature_matrix)