import csv
import grewpy
from grewpy import Corpus, CorpusDraft, Request


treebank_path = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/data/input/sud-treebanks-v2.16/SUD_French-Sequoia"
output_path = "sequoia_small_pos_removed.conllu"

corpus = Corpus(treebank_path)
draft = CorpusDraft(corpus)

for i in range(len(draft)):
        sentence = draft[i].features
        # deps = draft[i].sucs

        for token, feats in sentence.items():
            form = feats['form'] if 'form' in feats else ""
            lemma = feats['lemma'] if 'lemma' in feats else ""
            if 'upos' in feats:
                  if feats['upos'] not in ['NOUN', 'VERB', 'ADJ', 'PUNCT']:
                        sentence[token] = {'form': form, 'lemma': lemma, 'upos': 'NA'}
        draft[i].features = sentence

        # for token, deps in deps.items():
        #     print(f"Token: {token}, Deps: {deps}")

conll_string= draft.to_conll()
with open(output_path, "w", encoding="utf-8") as f:
    f.write("# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC\n")
    f.write(conll_string)