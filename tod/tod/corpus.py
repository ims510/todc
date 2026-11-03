import grewpy
import yaml
import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import json
from collections import Counter, defaultdict
from itertools import combinations
from grewpy import GRSDraft, GRS
sys.path.insert(1, "/Users/madalina/Documents/M2TAL/stage/grex/grex2")
import pyximport
import pandas as pd
pyximport.install()
import grex.data
import grex.utils
import grex.features


def remove_enhanced_deps(corpus):
    s = """
    package UD2bUD {
    rule enh { % remove enhanced relations
        pattern { e:N -[enhanced=yes]-> M }
        commands { del_edge e}
    }

    rule empty { % remove empty nodes
        pattern { N [wordform=__EMPTY__, textform=_] }
        commands { del_node N }
    }
    }

    strat main { Onf(UD2bUD) }
    """
    grs_draft = GRSDraft(s)
    grs = GRS(grs_draft)
    return corpus.apply(grs)

class Corpus:
    def __init__(
        self,
        treebank_path: str,
        grew_pattern: str,
        patterns_text_file: str,
        use_sud: bool = False,
        min_occurrences: int = 10,
        matrix_type: str = "PMI",
        excluded_feature_patterns: list = None,
        included_feature_patterns: list = None
    ):
        if use_sud==True:
            grewpy.set_config("sud")
        else:
            grewpy.set_config("ud")

        corpus = remove_enhanced_deps(grewpy.Corpus(treebank_path))
        draft = grewpy.CorpusDraft(corpus)

        # all_matches is a dict with keys as matching nodes and values as a list of dictionaries of this kind of structure: 'chat': [{'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}]
        # for me this is because i clustered by lemma, so i get lemmas as keys
        all_matches = corpus.search(
            grewpy.Request(grew_pattern)
            .without("X[InIdiom=Yes]")
            .without("X[Idiom=Yes]")
            .without("X[InTitle=Yes]")
            .without("X[Title=Yes]")
            .without("X[Scrap=Yes]")
            .without("X[Foreign]")
            .without("X[Lang]")
            .without("X-[fixed]->Y")
            .without("Y-[flat:name]->X")
            .without("Y-[goeswith]->X"),
            clustering_keys=["X.lemma"],
        ) # getting rid of noisy matches

        matches = {}
        for key, value in all_matches.items(): # type: ignore
            # remove those that have less than 10 occurrences
            if len(value) > min_occurrences:
                matches[key] = value

        # Create a dictionary to map sent_id to sentences for quick lookup
        # So for each sent_id I have each node and its features:  {'fr-ud-train_00001': {'0': {'form': '__0__'},'1': {'Definite': 'Def','Number': 'Plur', 'form': 'Les'}, '2': {'Gender': 'Fem','Number': 'Plur','form': 'commotions',}} etc
        sent_id_to_sentence = {
            draft[i].meta["sent_id"]: draft[i].features for i in range(len(draft)) # type: ignore
        }

        match_upos = {}
        # iterating through each lemma and its values like this: '€': [{'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}]
        for key, value in matches.items():
            # iterating through each match for the lemma so for example {'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}
            for m in value:
                match_sent_id = m["sent_id"]
                match_node_index = str(m["matching"]["nodes"]["X"])
                if match_sent_id in sent_id_to_sentence:
                    current_sentence_features = sent_id_to_sentence[match_sent_id]
                    if match_node_index in current_sentence_features.keys():
                        current_token_features = current_sentence_features[
                            match_node_index
                        ]
                        # if the token has a feature called ExtPos, we use that as the key, otherwise we use the upos feature
                        # ExtPos is used for example in 10% -> % has upos SYM but ExtPos Noun , so we want to use noun
                        # and what we're doing is creating a dictionary with keys (lemma, pos) and values as a list of matches
                        # so for example ('reason', 'NOUN') -> [match1, match2, match3] where a match looks like this: {'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}
                       
                        if "ExtPos" in current_token_features:
                            match_upos.setdefault(
                                (key, current_token_features["ExtPos"]), []
                            ).append(m)
                        else:
                            try:
                                match_upos.setdefault(
                                    (key, current_token_features["upos"]), []
                                ).append(m)
                            except KeyError:
                                print(f"WARNING: Token with lemma '{key}' in sentence '{match_sent_id}' does not have a 'upos' feature.")
                            #     match_upos.setdefault(
                            #         (key, "NA"), []
                            #     ).append(m)

        # remove those that have less than 10 occurrences again because now we've split by pos so we might have some that are less than 10 
        # for example if we had reason with 11 occurrences before and now we have reason:VERB with 5 occurrences and reason:NOUN with 6 occurrences
        new_match_upos = {}
        for key, value in match_upos.items():
            if len(value) > min_occurrences:
                new_match_upos[key] = value
        match_upos = new_match_upos

        for key, value in match_upos.items():
            if len(value) < min_occurrences:
                print(f"WARNING: {key} has only {len(value)} occurrences but passed filtering!")
        
        # grex stuff
        with open(patterns_text_file) as in_stream:
            config = yaml.load(in_stream, Loader=yaml.Loader)

        templates = grex.utils.FeaturePredicate.from_config(config["templates"])
        feature_predicate = grex.utils.FeaturePredicate.from_config(
            config["features"], templates=templates
        )

        # data is a dict where each (lemma, pos) is a key and the value is an empty list for now
        data = {k: list() for k in match_upos}
        for lex_unit, mts in match_upos.items():
            for match in mts:
                features = grex.data.extract_features(draft, match, feature_predicate)
                # features looks like this: features={('node', 'X', 'own', 'Number'): 'Plur', ('node', 'X', 'own', 'rel_shallow'): 'comp:obj'
                formatted_features = [
                    (
                        f"{':'.join(k)}={v}"
                        if not isinstance(v, set) # if the value is a set
                        else f"{':'.join(k)}={val}" # if the value is a set, we want to iterate through it and create a string for each value
                    )
                    for k, v in features.items()
                    for val in (v if isinstance(v, set) else [v])
                ]
                # so what we did is go from features = {('key1', 'key2'): 'value1', ('key3',): {'value2', 'value3'}} to ['key1:key2=value1', 'key3=value2', 'key3=value3']
                # and now we append the features to the data dict for the corresponding (lemma, pos) key
                if excluded_feature_patterns or included_feature_patterns:
                    formatted_features = self._filter_features(formatted_features, excluded_patterns=excluded_feature_patterns, included_patterns=included_feature_patterns)
                data[lex_unit].append(formatted_features)

        unique_lemma = sorted(set([k for k in data]))
        unique_features = sorted(
            set(
                [
                    feat
                    for _, match_upos in data.items() # iterating through (lemma, pos) : list of features for each match
                    for m in match_upos # iterating through each feature for the match
                    for feat in m
                ]
            )
        )

        self._idx2feature = {i: feat for i, feat in enumerate(unique_features)}
        self._feature2idx = {feat: i for i, feat in self._idx2feature.items()}
        self._idx2lexunit = {i: feat for i, feat in enumerate(unique_lemma)}
        self._lexunit2idx = {feat: i for i, feat in self._idx2lexunit.items()}
        self._lexunit_counts = {self._lexunit2idx[lu]: len(feats) for lu, feats in data.items()}

        self.co_occurrence_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        for lexunit, feature_lists in data.items():
            for feature_list in feature_lists:
                for feature in feature_list:
                    self.co_occurrence_matrix[self._lexunit2idx[lexunit], self._feature2idx[feature]] += 1

        if matrix_type == "precision":
            self.make_precision_matrix(self.co_occurrence_matrix)
        elif matrix_type == "coverage":
            self.make_coverage_matrix(self.co_occurrence_matrix)
        elif matrix_type == "PMI":
            self.make_pmi_matrix(self.co_occurrence_matrix)
        elif matrix_type == "tf-idf":
            self.make_tfidf_matrix(self.co_occurrence_matrix)
        elif matrix_type == "geometric_mean":
            self.make_geometric_mean_matrix(self.co_occurrence_matrix)
        else:
            raise ValueError("Invalid matrix type. Choose from 'precision', 'coverage', 'PMI', or 'tf-idf'.")
    
        corpus.clean()

    def idx2feature(self, idx: int) -> str:
        """Returns the feature name for a given index."""
        return self._idx2feature[idx]
    
    def feature2idx(self, feature: str) -> int:
        """Returns the index for a given feature name."""
        return self._feature2idx[feature]
    
    def idx2lexunit(self, idx: int) -> str:
        """Returns the lex unit name for a given index."""
        return self._idx2lexunit[idx]
    
    def lexunit2idx(self, lexunit: str) -> int:
        """Returns the index for a given lex unit name."""
        return self._lexunit2idx[lexunit]
    
    def make_precision_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the precision calculation: occurrences of word 1 with feature 1 / occurrences of all words with feature 1"""
        self.precision_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        column_sums = co_occurrence_matrix.sum(axis=0)
        self.precision_matrix = co_occurrence_matrix / column_sums
        # Avoid division by zero
        self.precision_matrix[np.isnan(self.precision_matrix)] = 0
        self.feature_matrix = self.precision_matrix

    def make_coverage_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the coverage calculation: occurrences of word 1 with feature 1 / occurrences of word 1"""
        self.coverage_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        self.coverage_matrix = co_occurrence_matrix / row_sums[:, np.newaxis]
        # Avoid division by zero
        self.coverage_matrix[np.isnan(self.coverage_matrix)] = 0
        self.feature_matrix = self.coverage_matrix

    def make_geometric_mean_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the geometric mean calculation: sqrt(precision * coverage)"""
        self.geometric_mean_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        self.precision_matrix = co_occurrence_matrix / co_occurrence_matrix.sum(axis=0)
        self.coverage_matrix = co_occurrence_matrix / co_occurrence_matrix.sum(axis=1)[:, np.newaxis]
        # Avoid division by zero
        self.precision_matrix[np.isnan(self.precision_matrix)] = 0
        self.coverage_matrix[np.isnan(self.coverage_matrix)] = 0
        self.geometric_mean_matrix = np.sqrt(
            self.precision_matrix * self.coverage_matrix
        )
        # Avoid division by zero
        self.feature_matrix = self.geometric_mean_matrix

    def make_pmi_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the PMI calculation: log( w1 with feature 1 * sum of co-occurrence matrix / sum of line * sum of column)"""
        self.pmi_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        column_sums = co_occurrence_matrix.sum(axis=0)
        total_sum = co_occurrence_matrix.sum()
        for i in range(len(self._idx2lexunit)):
            for j in range(len(self._idx2feature)):
                if co_occurrence_matrix[i, j] > 0:
                    self.pmi_matrix[i, j] = np.log(
                        (co_occurrence_matrix[i, j] * total_sum) /
                        (row_sums[i] * column_sums[j])
                    )
                else:
                    self.pmi_matrix[i, j] = 0  # Avoid log(0)

        self.feature_matrix = self.pmi_matrix
    
    def make_tfidf_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the tf-idf calculation: tf = occurrences of word 1 with feature 1 / occurrences of all other features for word 1 and idf = log(total number of lexical units / the number of lexical units with feature 1)"""
        self.tfidf_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        total_lexunits = co_occurrence_matrix.shape[0]
        column_sums = co_occurrence_matrix.sum(axis=0)
        for i in range(len(self._idx2lexunit)):
            for j in range(len(self._idx2feature)):
                if co_occurrence_matrix[i, j] > 0:
                    tf = co_occurrence_matrix[i, j] / row_sums[i]
                    idf = np.log(total_lexunits / column_sums[j])
                    self.tfidf_matrix[i, j] = tf * idf
                else:
                    self.tfidf_matrix[i, j] = 0
        self.feature_matrix = self.tfidf_matrix
        
    def show_variance_graphs(self):
        self.make_coverage_matrix(self.co_occurrence_matrix)
        X_coverage = self.feature_matrix

        self.make_precision_matrix(self.co_occurrence_matrix)
        X_precision = self.feature_matrix

        self.make_pmi_matrix(self.co_occurrence_matrix)
        X_pmi = self.feature_matrix

        self.make_geometric_mean_matrix(self.co_occurrence_matrix)
        X_geometric_mean = self.feature_matrix

        self.make_tfidf_matrix(self.co_occurrence_matrix)
        X_tfidf = self.feature_matrix

        # Calculate variance for each column in the matrices
        variances = {
            "Coverage": np.var(X_coverage, axis=0),
            "Precision": np.var(X_precision, axis=0),
            "PMI": np.var(X_pmi, axis=0),
            "TF-IDF": np.var(X_tfidf, axis=0),
            "Geometric Mean": np.var(X_geometric_mean, axis=0),
        }

        # Create subplots
        fig = make_subplots(rows=5, cols=1, subplot_titles=list(variances.keys()), shared_xaxes=True)

        # Add bar plots for each matrix
        for i, (name, var) in enumerate(variances.items(), start=1):
            sorted_indices = np.argsort(var)  # Get indices that would sort the variance
            sorted_variances = var[sorted_indices]  # Sort variances
            sorted_features = [self.idx2feature(idx) for idx in sorted_indices]  # Map indices to features
            fig.add_trace(
                go.Bar(
                    x=list(range(len(sorted_variances))),
                    y=sorted_variances,
                    text=sorted_features,  # Feature names
                    hovertemplate="Feature: %{text}<br>Variance: %{y}<extra></extra>",
                ),
                row=i,
                col=1,
            )

        # Update layout
        fig.update_layout(
            height=1500,  # Adjust height for better visualization
            title="Variance per Column for Each Matrix",
            showlegend=False,
        )

        # Show plot
        fig.show()

        return X_coverage, X_precision, X_pmi, X_geometric_mean, X_tfidf
    
    def _filter_features(self, features, excluded_patterns=None, included_patterns=None):
        """Filter out features matching any of the excluded patterns or keep only features matching any of the included patterns."""
        import re
        filtered_features = []
        for feature in features:
            # check exclusion patterns
            if excluded_patterns:
                should_exclude = False
                for pattern in excluded_patterns:
                    if re.search(pattern, feature):
                        should_exclude = True
                        break
                if should_exclude:
                    continue
           # check inclusion patterns
            if included_patterns:
                should_include = False
                for pattern in included_patterns:
                    if re.search(pattern, feature):
                        should_include = True
                        break
                if not should_include:
                    continue    
            filtered_features.append(feature)
        return filtered_features

class CorpusDepClustering:
    def __init__(
        self,
        treebank_path: str,
        grew_pattern: str,
        patterns_text_file: str,
        use_sud: bool = False,
        min_occurrences: int = 10,
        min_dep_occurrences: int = 0,   # NEW: 0 disables the filter
        matrix_type: str = "PMI",
        excluded_feature_patterns: list = None,
        included_feature_patterns: list = None
    ):
        if use_sud == True:
            grewpy.set_config("sud")
        else:
            grewpy.set_config("ud")

        corpus = remove_enhanced_deps(grewpy.Corpus(treebank_path))
        draft = grewpy.CorpusDraft(corpus)

        # all_matches is now nested two levels deep:
        # {lemma: {edge_label_json: [list of matches]}}
        # e.g. '€': {'{"1":"obl","2":"mod"}': [{'sent_id': ..., 'matching': {...}}, ...]}
        all_matches = corpus.search(
            grewpy.Request(grew_pattern)
            .without("X[InIdiom=Yes]")
            .without("X[Idiom=Yes]")
            .without("X[InTitle=Yes]")
            .without("X[Title=Yes]")
            .without("X[Scrap=Yes]")
            .without("X[Foreign]")
            .without("X[Lang]")
            .without("X-[fixed]->Y")
            .without("Y-[flat:name]->X")
            .without("Y-[goeswith]->X"),
            clustering_parameter=["X.lemma", "e.label"],
        )

        # Raw count across all lemmas. A relation that exists only via
        # one or two noisy annotations gets dropped here before it can
        # ever enter the feature matrix.
        dep_relation_totals = {}
        for lemma, dep_dict in all_matches.items():  # type: ignore
            for edge_label_json, match_list in dep_dict.items():
                dep_relation = self._parse_edge_label(edge_label_json)
                dep_relation_totals[dep_relation] = (
                    dep_relation_totals.get(dep_relation, 0) + len(match_list)
                )

        rare_deps = {
            dep for dep, count in dep_relation_totals.items()
            if count < min_dep_occurrences
        }
        if rare_deps:
            print(f"Dropping {len(rare_deps)} rare dep relations "
                  f"(< {min_dep_occurrences} total occurrences): "
                  f"{sorted(rare_deps)}")
        # Flatten the nested structure into {(lemma, dep_relation): [matches]}
        # and apply the min_occurrences filter at this level so a lemma that's
        # frequent overall but rare under a given relation gets dropped.
        matches = {}
        for lemma, dep_dict in all_matches.items():  # type: ignore
            for edge_label_json, match_list in dep_dict.items():
                if len(match_list) > min_occurrences:
                    dep_relation = self._parse_edge_label(edge_label_json)
                    matches[(lemma, dep_relation)] = match_list

        sent_id_to_sentence = {
            draft[i].meta["sent_id"]: draft[i].features for i in range(len(draft))  # type: ignore
        }

        # match_upos now keyed by (lemma, pos, dep_relation) instead of (lemma, pos)
        match_upos = {}
        for (lemma, dep_relation), value in matches.items():
            for m in value:
                match_sent_id = m["sent_id"]
                match_node_index = str(m["matching"]["nodes"]["X"])
                if match_sent_id in sent_id_to_sentence:
                    current_sentence_features = sent_id_to_sentence[match_sent_id]
                    if match_node_index in current_sentence_features.keys():
                        current_token_features = current_sentence_features[match_node_index]
                        if "ExtPos" in current_token_features:
                            match_upos.setdefault(
                                (lemma, current_token_features["ExtPos"], dep_relation), []
                            ).append(m)
                        else:
                            try:
                                match_upos.setdefault(
                                    (lemma, current_token_features["upos"], dep_relation), []
                                ).append(m)
                            except KeyError:
                                print(f"WARNING: Token with lemma '{lemma}' in sentence "
                                      f"'{match_sent_id}' does not have a 'upos' feature.")

        # Re-filter: splitting by pos may have pushed some buckets under threshold.
        # (We already filtered before splitting by pos, but the same logic from the
        # original applies — a (lemma, dep) group might split across two POS tags.)
        new_match_upos = {}
        for key, value in match_upos.items():
            if len(value) > min_occurrences:
                new_match_upos[key] = value
        match_upos = new_match_upos

        for key, value in match_upos.items():
            if len(value) < min_occurrences:
                print(f"WARNING: {key} has only {len(value)} occurrences but passed filtering!")

        # ----- everything below is unchanged from the original -----

        with open(patterns_text_file) as in_stream:
            config = yaml.load(in_stream, Loader=yaml.Loader)

        templates = grex.utils.FeaturePredicate.from_config(config["templates"])
        feature_predicate = grex.utils.FeaturePredicate.from_config(
            config["features"], templates=templates
        )

        data = {k: list() for k in match_upos}
        for lex_unit, mts in match_upos.items():
            for match in mts:
                features = grex.data.extract_features(draft, match, feature_predicate)
                formatted_features = [
                    (
                        f"{':'.join(k)}={v}"
                        if not isinstance(v, set)
                        else f"{':'.join(k)}={val}"
                    )
                    for k, v in features.items()
                    for val in (v if isinstance(v, set) else [v])
                ]
                if excluded_feature_patterns or included_feature_patterns:
                    formatted_features = self._filter_features(
                        formatted_features,
                        excluded_patterns=excluded_feature_patterns,
                        included_patterns=included_feature_patterns,
                    )
                data[lex_unit].append(formatted_features)

        unique_lexunit = sorted(set([k for k in data]))
        unique_features = sorted(
            set(
                [
                    feat
                    for _, match_upos in data.items()
                    for m in match_upos
                    for feat in m
                ]
            )
        )

        self._idx2feature = {i: feat for i, feat in enumerate(unique_features)}
        self._feature2idx = {feat: i for i, feat in self._idx2feature.items()}
        self._idx2lexunit = {i: feat for i, feat in enumerate(unique_lexunit)}
        self._lexunit2idx = {feat: i for i, feat in self._idx2lexunit.items()}

        self.co_occurrence_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        for lexunit, feature_lists in data.items():
            for feature_list in feature_lists:
                for feature in feature_list:
                    self.co_occurrence_matrix[
                        self._lexunit2idx[lexunit], self._feature2idx[feature]
                    ] += 1
        if matrix_type == "precision":
            self.make_precision_matrix(self.co_occurrence_matrix)
        elif matrix_type == "coverage":
            self.make_coverage_matrix(self.co_occurrence_matrix)
        elif matrix_type == "PMI":
            self.make_pmi_matrix(self.co_occurrence_matrix)
        elif matrix_type == "tf-idf":
            self.make_tfidf_matrix(self.co_occurrence_matrix)
        elif matrix_type == "geometric_mean":
            self.make_geometric_mean_matrix(self.co_occurrence_matrix)
        else:
            raise ValueError("Invalid matrix type. Choose from 'precision', 'coverage', 'PMI', or 'tf-idf'.")
    
        corpus.clean()

    @staticmethod
    def _parse_edge_label(label_json: str) -> str:
        """Convert grew's edge label JSON string into a readable relation string.

        grew represents UD/SUD labels as a JSON dict where keys "1", "2", "3"
        encode the level (relation, subtype, sub-subtype). We re-join them with
        ':' to recover the standard surface form.

        Examples:
            '{"1":"obl","2":"mod"}'  -> 'obl:mod'
            '{"1":"nsubj"}'          -> 'nsubj'
            '{"1":"comp","2":"obj"}' -> 'comp:obj'
            '"appos"'                -> 'appos'
        """
        parsed = json.loads(label_json)
        
        # If it's a simple string (e.g., '"appos"'), json.loads returns str, not dict
        if isinstance(parsed, str):
            return parsed
        
        # Otherwise it's a dict with numeric keys "1", "2", etc.
        sorted_keys = sorted(parsed.keys(), key=int)
        return ":".join(parsed[k] for k in sorted_keys)


    def idx2feature(self, idx: int) -> str:
        """Returns the feature name for a given index."""
        return self._idx2feature[idx]
    
    def feature2idx(self, feature: str) -> int:
        """Returns the index for a given feature name."""
        return self._feature2idx[feature]
    
    def idx2lexunit(self, idx: int) -> str:
        """Returns the lex unit name for a given index."""
        return self._idx2lexunit[idx]
    
    def lexunit2idx(self, lexunit: str) -> int:
        """Returns the index for a given lex unit name."""
        return self._lexunit2idx[lexunit]
    
    def make_precision_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the precision calculation: occurrences of word 1 with feature 1 / occurrences of all words with feature 1"""
        self.precision_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        column_sums = co_occurrence_matrix.sum(axis=0)
        self.precision_matrix = co_occurrence_matrix / column_sums
        # Avoid division by zero
        self.precision_matrix[np.isnan(self.precision_matrix)] = 0
        self.feature_matrix = self.precision_matrix

    def make_coverage_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the coverage calculation: occurrences of word 1 with feature 1 / occurrences of word 1"""
        self.coverage_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        self.coverage_matrix = co_occurrence_matrix / row_sums[:, np.newaxis]
        # Avoid division by zero
        self.coverage_matrix[np.isnan(self.coverage_matrix)] = 0
        self.feature_matrix = self.coverage_matrix

    def make_geometric_mean_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the geometric mean calculation: sqrt(precision * coverage)"""
        self.geometric_mean_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        self.precision_matrix = co_occurrence_matrix / co_occurrence_matrix.sum(axis=0)
        self.coverage_matrix = co_occurrence_matrix / co_occurrence_matrix.sum(axis=1)[:, np.newaxis]
        # Avoid division by zero
        self.precision_matrix[np.isnan(self.precision_matrix)] = 0
        self.coverage_matrix[np.isnan(self.coverage_matrix)] = 0
        self.geometric_mean_matrix = np.sqrt(
            self.precision_matrix * self.coverage_matrix
        )
        # Avoid division by zero
        self.feature_matrix = self.geometric_mean_matrix

    def make_pmi_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the PMI calculation: log( w1 with feature 1 * sum of co-occurrence matrix / sum of line * sum of column)"""
        self.pmi_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        column_sums = co_occurrence_matrix.sum(axis=0)
        total_sum = co_occurrence_matrix.sum()
        for i in range(len(self._idx2lexunit)):
            for j in range(len(self._idx2feature)):
                if co_occurrence_matrix[i, j] > 0:
                    self.pmi_matrix[i, j] = np.log(
                        (co_occurrence_matrix[i, j] * total_sum) /
                        (row_sums[i] * column_sums[j])
                    )
                else:
                    self.pmi_matrix[i, j] = 0  # Avoid log(0)

        self.feature_matrix = self.pmi_matrix
    
    def make_tfidf_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the tf-idf calculation: tf = occurrences of word 1 with feature 1 / occurrences of all other features for word 1 and idf = log(total number of lexical units / the number of lexical units with feature 1)"""
        self.tfidf_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        total_lexunits = co_occurrence_matrix.shape[0]
        column_sums = co_occurrence_matrix.sum(axis=0)
        for i in range(len(self._idx2lexunit)):
            for j in range(len(self._idx2feature)):
                if co_occurrence_matrix[i, j] > 0:
                    tf = co_occurrence_matrix[i, j] / row_sums[i]
                    idf = np.log(total_lexunits / column_sums[j])
                    self.tfidf_matrix[i, j] = tf * idf
                else:
                    self.tfidf_matrix[i, j] = 0
        self.feature_matrix = self.tfidf_matrix
        
    def show_variance_graphs(self):
        self.make_coverage_matrix(self.co_occurrence_matrix)
        X_coverage = self.feature_matrix

        self.make_precision_matrix(self.co_occurrence_matrix)
        X_precision = self.feature_matrix

        self.make_pmi_matrix(self.co_occurrence_matrix)
        X_pmi = self.feature_matrix

        self.make_geometric_mean_matrix(self.co_occurrence_matrix)
        X_geometric_mean = self.feature_matrix

        self.make_tfidf_matrix(self.co_occurrence_matrix)
        X_tfidf = self.feature_matrix

        # Calculate variance for each column in the matrices
        variances = {
            "Coverage": np.var(X_coverage, axis=0),
            "Precision": np.var(X_precision, axis=0),
            "PMI": np.var(X_pmi, axis=0),
            "TF-IDF": np.var(X_tfidf, axis=0),
            "Geometric Mean": np.var(X_geometric_mean, axis=0),
        }

        # Create subplots
        fig = make_subplots(rows=5, cols=1, subplot_titles=list(variances.keys()), shared_xaxes=True)

        # Add bar plots for each matrix
        for i, (name, var) in enumerate(variances.items(), start=1):
            sorted_indices = np.argsort(var)  # Get indices that would sort the variance
            sorted_variances = var[sorted_indices]  # Sort variances
            sorted_features = [self.idx2feature(idx) for idx in sorted_indices]  # Map indices to features
            fig.add_trace(
                go.Bar(
                    x=list(range(len(sorted_variances))),
                    y=sorted_variances,
                    text=sorted_features,  # Feature names
                    hovertemplate="Feature: %{text}<br>Variance: %{y}<extra></extra>",
                ),
                row=i,
                col=1,
            )

        # Update layout
        fig.update_layout(
            height=1500,  # Adjust height for better visualization
            title="Variance per Column for Each Matrix",
            showlegend=False,
        )

        # Show plot
        fig.show()

        return X_coverage, X_precision, X_pmi, X_geometric_mean, X_tfidf
    
    def _filter_features(self, features, excluded_patterns=None, included_patterns=None):
        """Filter out features matching any of the excluded patterns or keep only features matching any of the included patterns."""
        import re
        filtered_features = []
        for feature in features:
            # check exclusion patterns
            if excluded_patterns:
                should_exclude = False
                for pattern in excluded_patterns:
                    if re.search(pattern, feature):
                        should_exclude = True
                        break
                if should_exclude:
                    continue
           # check inclusion patterns
            if included_patterns:
                should_include = False
                for pattern in included_patterns:
                    if re.search(pattern, feature):
                        should_include = True
                        break
                if not should_include:
                    continue    
            filtered_features.append(feature)
        return filtered_features

class CorpusNoLemma:
    def __init__(
        self,
        treebank_path: str,
        grew_pattern: str,
        patterns_text_file: str,
        use_sud: bool = False,
        min_occurrences: int = 10,
        matrix_type: str = "PMI",
        excluded_feature_patterns: list = None,
        included_feature_patterns: list = None
    ):
        if use_sud==True:
            grewpy.set_config("sud")
        else:
            grewpy.set_config("ud")

        corpus = remove_enhanced_deps(grewpy.Corpus(treebank_path))
        draft = grewpy.CorpusDraft(corpus)

        # all_matches is a dict with keys as matching nodes and values as a list of dictionaries of this kind of structure: 'chat': [{'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}]
        # for me this is because i clustered by lemma, so i get lemmas as keys
        all_matches = corpus.search(
            grewpy.Request(grew_pattern)
            .without("X[InIdiom=Yes]")
            .without("X[Idiom=Yes]")
            .without("X[InTitle=Yes]")
            .without("X[Title=Yes]")
            .without("X[Scrap=Yes]")
            .without("X[Foreign]")
            .without("X[Lang]")
            .without("X-[fixed]->Y")
            .without("Y-[flat:name]->X")
            .without("Y-[goeswith]->X"),
        ) # getting rid of noisy matches

        # Create a dictionary to map sent_id to sentences for quick lookup
        sent_id_to_sentence = {
            draft[i].meta["sent_id"]: draft[i].features for i in range(len(draft)) # type: ignore
        }

        match_upos = {}

        for m in all_matches:
            match_sent_id = m["sent_id"]
            match_node_index = str(m["matching"]["nodes"]["X"])
            
            if match_sent_id in sent_id_to_sentence:
                current_sentence_features = sent_id_to_sentence[match_sent_id]
                
                if match_node_index in current_sentence_features:
                    current_token_features = current_sentence_features[match_node_index]
                    
                    # --- THE FALLBACK LOGIC ---
                    # Check for lemma. If it's missing, None, or CoNLL's empty "_", fallback to form.
                    lex_unit = current_token_features.get("lemma")
                    if not lex_unit or lex_unit == "_":
                        lex_unit = current_token_features.get("form", "UNKNOWN")
                    
                    # Check for ExtPos. If missing, fallback to upos.
                    pos_unit = current_token_features.get("ExtPos")
                    if not pos_unit:
                        pos_unit = current_token_features.get("upos")
                    
                    # Group by the new (lex_unit, pos_unit) tuple
                    if pos_unit:
                        match_upos.setdefault((lex_unit, pos_unit), []).append(m)
                    else:
                        print(f"WARNING: Token '{lex_unit}' in sentence '{match_sent_id}' lacks a POS feature.")

        # 3. Filter for minimum occurrences JUST ONCE
        new_match_upos = {}
        for key, value in match_upos.items():
            # Note: I changed > to >= so that `min_occurrences=10` keeps items with exactly 10.
            if len(value) >= min_occurrences: 
                new_match_upos[key] = value
        
        match_upos = new_match_upos

        for key, value in match_upos.items():
            if len(value) < min_occurrences:
                print(f"WARNING: {key} has only {len(value)} occurrences but passed filtering!")
        
        # grex stuff
        with open(patterns_text_file) as in_stream:
            config = yaml.load(in_stream, Loader=yaml.Loader)

        templates = grex.utils.FeaturePredicate.from_config(config["templates"])
        feature_predicate = grex.utils.FeaturePredicate.from_config(
            config["features"], templates=templates
        )

        # data is a dict where each (lemma, pos) is a key and the value is an empty list for now
        data = {k: list() for k in match_upos}
        for lex_unit, mts in match_upos.items():
            for match in mts:
                features = grex.data.extract_features(draft, match, feature_predicate)
                # features looks like this: features={('node', 'X', 'own', 'Number'): 'Plur', ('node', 'X', 'own', 'rel_shallow'): 'comp:obj'
                formatted_features = [
                    (
                        f"{':'.join(k)}={v}"
                        if not isinstance(v, set) # if the value is a set
                        else f"{':'.join(k)}={val}" # if the value is a set, we want to iterate through it and create a string for each value
                    )
                    for k, v in features.items()
                    for val in (v if isinstance(v, set) else [v])
                ]
                # so what we did is go from features = {('key1', 'key2'): 'value1', ('key3',): {'value2', 'value3'}} to ['key1:key2=value1', 'key3=value2', 'key3=value3']
                # and now we append the features to the data dict for the corresponding (lemma, pos) key
                if excluded_feature_patterns or included_feature_patterns:
                    formatted_features = self._filter_features(formatted_features, excluded_patterns=excluded_feature_patterns, included_patterns=included_feature_patterns)
                data[lex_unit].append(formatted_features)

        unique_lemma = sorted(set([k for k in data]))
        unique_features = sorted(
            set(
                [
                    feat
                    for _, match_upos in data.items() # iterating through (lemma, pos) : list of features for each match
                    for m in match_upos # iterating through each feature for the match
                    for feat in m
                ]
            )
        )

        self._idx2feature = {i: feat for i, feat in enumerate(unique_features)}
        self._feature2idx = {feat: i for i, feat in self._idx2feature.items()}
        self._idx2lexunit = {i: feat for i, feat in enumerate(unique_lemma)}
        self._lexunit2idx = {feat: i for i, feat in self._idx2lexunit.items()}

        self.co_occurrence_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        for lexunit, feature_lists in data.items():
            for feature_list in feature_lists:
                for feature in feature_list:
                    self.co_occurrence_matrix[self._lexunit2idx[lexunit], self._feature2idx[feature]] += 1

        if matrix_type == "precision":
            self.make_precision_matrix(self.co_occurrence_matrix)
        elif matrix_type == "coverage":
            self.make_coverage_matrix(self.co_occurrence_matrix)
        elif matrix_type == "PMI":
            self.make_pmi_matrix(self.co_occurrence_matrix)
        elif matrix_type == "tf-idf":
            self.make_tfidf_matrix(self.co_occurrence_matrix)
        elif matrix_type == "geometric_mean":
            self.make_geometric_mean_matrix(self.co_occurrence_matrix)
        else:
            raise ValueError("Invalid matrix type. Choose from 'precision', 'coverage', 'PMI', or 'tf-idf'.")
    
        corpus.clean()

    def idx2feature(self, idx: int) -> str:
        """Returns the feature name for a given index."""
        return self._idx2feature[idx]
    
    def feature2idx(self, feature: str) -> int:
        """Returns the index for a given feature name."""
        return self._feature2idx[feature]
    
    def idx2lexunit(self, idx: int) -> str:
        """Returns the lex unit name for a given index."""
        return self._idx2lexunit[idx]
    
    def lexunit2idx(self, lexunit: str) -> int:
        """Returns the index for a given lex unit name."""
        return self._lexunit2idx[lexunit]
    
    def make_precision_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the precision calculation: occurrences of word 1 with feature 1 / occurrences of all words with feature 1"""
        self.precision_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        column_sums = co_occurrence_matrix.sum(axis=0)
        self.precision_matrix = co_occurrence_matrix / column_sums
        # Avoid division by zero
        self.precision_matrix[np.isnan(self.precision_matrix)] = 0
        self.feature_matrix = self.precision_matrix

    def make_coverage_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the coverage calculation: occurrences of word 1 with feature 1 / occurrences of word 1"""
        self.coverage_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        self.coverage_matrix = co_occurrence_matrix / row_sums[:, np.newaxis]
        # Avoid division by zero
        self.coverage_matrix[np.isnan(self.coverage_matrix)] = 0
        self.feature_matrix = self.coverage_matrix

    def make_geometric_mean_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the geometric mean calculation: sqrt(precision * coverage)"""
        self.geometric_mean_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        self.precision_matrix = co_occurrence_matrix / co_occurrence_matrix.sum(axis=0)
        self.coverage_matrix = co_occurrence_matrix / co_occurrence_matrix.sum(axis=1)[:, np.newaxis]
        # Avoid division by zero
        self.precision_matrix[np.isnan(self.precision_matrix)] = 0
        self.coverage_matrix[np.isnan(self.coverage_matrix)] = 0
        self.geometric_mean_matrix = np.sqrt(
            self.precision_matrix * self.coverage_matrix
        )
        # Avoid division by zero
        self.feature_matrix = self.geometric_mean_matrix

    def make_pmi_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the PMI calculation: log( w1 with feature 1 * sum of co-occurrence matrix / sum of line * sum of column)"""
        self.pmi_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        column_sums = co_occurrence_matrix.sum(axis=0)
        total_sum = co_occurrence_matrix.sum()
        for i in range(len(self._idx2lexunit)):
            for j in range(len(self._idx2feature)):
                if co_occurrence_matrix[i, j] > 0:
                    self.pmi_matrix[i, j] = np.log(
                        (co_occurrence_matrix[i, j] * total_sum) /
                        (row_sums[i] * column_sums[j])
                    )
                else:
                    self.pmi_matrix[i, j] = 0  # Avoid log(0)

        self.feature_matrix = self.pmi_matrix
    
    def make_tfidf_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the tf-idf calculation: tf = occurrences of word 1 with feature 1 / occurrences of all other features for word 1 and idf = log(total number of lexical units / the number of lexical units with feature 1)"""
        self.tfidf_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        total_lexunits = co_occurrence_matrix.shape[0]
        column_sums = co_occurrence_matrix.sum(axis=0)
        for i in range(len(self._idx2lexunit)):
            for j in range(len(self._idx2feature)):
                if co_occurrence_matrix[i, j] > 0:
                    tf = co_occurrence_matrix[i, j] / row_sums[i]
                    idf = np.log(total_lexunits / column_sums[j])
                    self.tfidf_matrix[i, j] = tf * idf
                else:
                    self.tfidf_matrix[i, j] = 0
        self.feature_matrix = self.tfidf_matrix
        
    def show_variance_graphs(self):
        self.make_coverage_matrix(self.co_occurrence_matrix)
        X_coverage = self.feature_matrix

        self.make_precision_matrix(self.co_occurrence_matrix)
        X_precision = self.feature_matrix

        self.make_pmi_matrix(self.co_occurrence_matrix)
        X_pmi = self.feature_matrix

        self.make_geometric_mean_matrix(self.co_occurrence_matrix)
        X_geometric_mean = self.feature_matrix

        self.make_tfidf_matrix(self.co_occurrence_matrix)
        X_tfidf = self.feature_matrix

        # Calculate variance for each column in the matrices
        variances = {
            "Coverage": np.var(X_coverage, axis=0),
            "Precision": np.var(X_precision, axis=0),
            "PMI": np.var(X_pmi, axis=0),
            "TF-IDF": np.var(X_tfidf, axis=0),
            "Geometric Mean": np.var(X_geometric_mean, axis=0),
        }

        # Create subplots
        fig = make_subplots(rows=5, cols=1, subplot_titles=list(variances.keys()), shared_xaxes=True)

        # Add bar plots for each matrix
        for i, (name, var) in enumerate(variances.items(), start=1):
            sorted_indices = np.argsort(var)  # Get indices that would sort the variance
            sorted_variances = var[sorted_indices]  # Sort variances
            sorted_features = [self.idx2feature(idx) for idx in sorted_indices]  # Map indices to features
            fig.add_trace(
                go.Bar(
                    x=list(range(len(sorted_variances))),
                    y=sorted_variances,
                    text=sorted_features,  # Feature names
                    hovertemplate="Feature: %{text}<br>Variance: %{y}<extra></extra>",
                ),
                row=i,
                col=1,
            )

        # Update layout
        fig.update_layout(
            height=1500,  # Adjust height for better visualization
            title="Variance per Column for Each Matrix",
            showlegend=False,
        )

        # Show plot
        fig.show()

        return X_coverage, X_precision, X_pmi, X_geometric_mean, X_tfidf
    
    def _filter_features(self, features, excluded_patterns=None, included_patterns=None):
        """Filter out features matching any of the excluded patterns or keep only features matching any of the included patterns."""
        import re
        filtered_features = []
        for feature in features:
            # check exclusion patterns
            if excluded_patterns:
                should_exclude = False
                for pattern in excluded_patterns:
                    if re.search(pattern, feature):
                        should_exclude = True
                        break
                if should_exclude:
                    continue
           # check inclusion patterns
            if included_patterns:
                should_include = False
                for pattern in included_patterns:
                    if re.search(pattern, feature):
                        should_include = True
                        break
                if not should_include:
                    continue    
            filtered_features.append(feature)
        return filtered_features
    
class CorpusFromMatrix:
    def __init__(self, matrix_path, lexunits_path, feature_names_path, coocc_matrix_path=None):
        # Load matrix
        self.feature_matrix = np.load(matrix_path)

        # Load lexical units
        df = pd.read_csv(lexunits_path)
        lexunits = df["lexunit"].tolist()

        # Load feature names
        feature_names = np.load(feature_names_path)

        # Build indices
        self._idx2lexunit = {i: lu for i, lu in enumerate(lexunits)}
        self._lexunit2idx = {lu: i for i, lu in self._idx2lexunit.items()}

        self._idx2feature = {i: f for i, f in enumerate(feature_names)}
        self._feature2idx = {f: i for i, f in self._idx2feature.items()}

        if coocc_matrix_path is not None:
            self.co_occurrence_matrix = np.load(coocc_matrix_path)
        # Sanity checks
        if self.feature_matrix.shape != (len(lexunits), len(feature_names)):
            raise ValueError(
                f"Matrix shape {self.feature_matrix.shape} does not match "
                f"{len(lexunits)} lexunits x {len(feature_names)} features."
            )
        
    def idx2feature(self, idx: int) -> str:
        return self._idx2feature[idx]

    def feature2idx(self, feature: str) -> int:
        return self._feature2idx[feature]

    def idx2lexunit(self, idx: int) -> str:
        return self._idx2lexunit[idx]

    def lexunit2idx(self, lexunit: str) -> int:
        return self._lexunit2idx[lexunit]


    
class CorpusTriplet:
    """A class which implements the lexical units as triplets (lemma, pos, treebank)"""
    def __init__(
        self,
        treebank_path: str,
        grew_pattern: str,
        patterns_text_file: str,
        use_sud: bool = False,
        min_occurrences: int = 10,
        matrix_type: str = "PMI",
    ):
        if use_sud:
            grewpy.set_config("sud")
        else:
            grewpy.set_config("ud")

        corpus = grewpy.Corpus(treebank_path)
        draft = grewpy.CorpusDraft(corpus)

        # all_matches is a dict with keys as matching nodes and values as a list of dictionaries of this kind of structure: 'chat': [{'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}]
        # for me this is because i clustered by lemma, so i get lemmas as keys
        all_matches = corpus.search(
            grewpy.Request(grew_pattern)
            .without("X[InIdiom=Yes]")
            .without("X[Idiom=Yes]")
            .without("X[InTitle=Yes]")
            .without("X[Title=Yes]")
            .without("X[Scrap=Yes]")
            .without("X[Foreign]")
            .without("X[Lang]")
            .without("X-[fixed]->Y")
            .without("Y-[flat:name]->X")
            .without("Y-[goeswith]->X"),
            clustering_parameter=["X.lemma"],
        ) # getting rid of noisy matches

        matches = {}
        for key, value in all_matches.items(): # type: ignore
            # remove those that have less than 10 occurrences
            if len(value) > min_occurrences:
                matches[key] = value

        # Create a dictionary to map sent_id to sentences for quick lookup
        # So for each sent_id I have each node and its features:  {'fr-ud-train_00001': {'0': {'form': '__0__'},'1': {'Definite': 'Def','Number': 'Plur', 'form': 'Les'}, '2': {'Gender': 'Fem','Number': 'Plur','form': 'commotions',}} etc
        sent_id_to_sentence = {
            draft[i].meta["sent_id"]: draft[i].features for i in range(len(draft)) # type: ignore
        }

        match_upos = {}
        # iterating through each lemma and its values like this: '€': [{'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}]
        for key, value in matches.items():
            # iterating through each match for the lemma so for example {'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}
            for m in value:
                
                match_sent_id = m["sent_id"]
                
                match_node_index = str(m["matching"]["nodes"]["X"])
                if match_sent_id in sent_id_to_sentence:
                    current_sentence_features = sent_id_to_sentence[match_sent_id]
                    match_treebank = draft[match_sent_id].meta['treebank']
                    if match_node_index in current_sentence_features.keys():
                        current_token_features = current_sentence_features[
                            match_node_index
                        ]
                        # if the token has a feature called ExtPos, we use that as the key, otherwise we use the upos feature
                        # ExtPos is used for example in 10% -> % has upos SYM but ExtPos Noun , so we want to use noun
                        # and what we're doing is creating a dictionary with keys (lemma, pos) and values as a list of matches
                        # so for example ('reason', 'NOUN') -> [match1, match2, match3] where a match looks like this: {'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}
                       
                        if "ExtPos" in current_token_features:
                            match_upos.setdefault(
                                (key, current_token_features["ExtPos"], match_treebank), []
                            ).append(m)
                        else:
                            try:
                                match_upos.setdefault(
                                    (key, current_token_features["upos"], match_treebank), []
                                ).append(m)
                            except KeyError:
                                print(f"WARNING: Token with lemma '{key}' in sentence '{match_sent_id}' does not have a 'upos' feature.")
                            #     match_upos.setdefault(
                            #         (key, "NA"), []
                            #     ).append(m)

        # remove those that have less than 10 occurrences again because now we've split by pos so we might have some that are less than 10 
        # for example if we had reason with 11 occurrences before and now we have reason:VERB with 5 occurrences and reason:NOUN with 6 occurrences
        new_match_upos = {}
        for key, value in match_upos.items():
            if len(value) > min_occurrences:
                new_match_upos[key] = value
        match_upos = new_match_upos

        for key, value in match_upos.items():
            if len(value) < min_occurrences:
                print(f"WARNING: {key} has only {len(value)} occurrences but passed filtering!")
        # UNCOMMENT THIS if comparing treebanks of the same language so you want to compare the same word-pos pairs
        # pair_counts = Counter(k[:2] for k in match_upos.keys())
        # new_dict = {k: v for k, v in match_upos.items() if pair_counts[k[:2]] >= 2}
        # match_upos = new_dict
        # print(match_upos.keys())
        # grex stuff
        with open(patterns_text_file) as in_stream:
            config = yaml.load(in_stream, Loader=yaml.Loader)

        templates = grex.utils.FeaturePredicate.from_config(config["templates"])
        feature_predicate = grex.utils.FeaturePredicate.from_config(
            config["features"], templates=templates
        )

        # data is a dict where each (lemma, pos) is a key and the value is an empty list for now
        data = {k: list() for k in match_upos}
        for lex_unit, mts in match_upos.items():
            for match in mts:
                try:
                    features = grex.data.extract_features(draft, match, feature_predicate)
                except AssertionError:
                    print(f"WARNING: Skipping sent_id={match.get('sent_id')}")
                    continue
                # features looks like this: features={('node', 'X', 'own', 'Number'): 'Plur', ('node', 'X', 'own', 'rel_shallow'): 'comp:obj'
                formatted_features = [
                    (
                        f"{':'.join(k)}={v}"
                        if not isinstance(v, set) # if the value is a set
                        else f"{':'.join(k)}={val}" # if the value is a set, we want to iterate through it and create a string for each value
                    )
                    for k, v in features.items()
                    for val in (v if isinstance(v, set) else [v])
                ]
                # so what we did is go from features = {('key1', 'key2'): 'value1', ('key3',): {'value2', 'value3'}} to ['key1:key2=value1', 'key3=value2', 'key3=value3']
                # and now we append the features to the data dict for the corresponding (lemma, pos) key
                if excluded_feature_patterns:
                    formatted_features = self._filter_features(formatted_features, excluded_feature_patterns)
                data[lex_unit].append(formatted_features)

        unique_lemma = sorted(set([k for k in data]))
        unique_features = sorted(
            set(
                [
                    feat
                    for _, match_upos in data.items() # iterating through (lemma, pos) : list of features for each match
                    for m in match_upos # iterating through each feature for the match
                    for feat in m
                ]
            )
        )

        self._idx2feature = {i: feat for i, feat in enumerate(unique_features)}
        self._feature2idx = {feat: i for i, feat in self._idx2feature.items()}
        self._idx2lexunit = {i: feat for i, feat in enumerate(unique_lemma)}
        self._lexunit2idx = {feat: i for i, feat in self._idx2lexunit.items()}

        self.co_occurrence_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        for lexunit, feature_lists in data.items():
            for feature_list in feature_lists:
                for feature in feature_list:
                    self.co_occurrence_matrix[self._lexunit2idx[lexunit], self._feature2idx[feature]] += 1

        if matrix_type == "precision":
            self.make_precision_matrix(self.co_occurrence_matrix)
        elif matrix_type == "coverage":
            self.make_coverage_matrix(self.co_occurrence_matrix)
        elif matrix_type == "PMI":
            self.make_pmi_matrix(self.co_occurrence_matrix)
        elif matrix_type == "tf-idf":
            self.make_tfidf_matrix(self.co_occurrence_matrix)
        elif matrix_type == "geometric_mean":
            self.make_geometric_mean_matrix(self.co_occurrence_matrix)
        else:
            raise ValueError("Invalid matrix type. Choose from 'precision', 'coverage', 'PMI', or 'tf-idf'.")


    def idx2feature(self, idx: int) -> str:
        """Returns the feature name for a given index."""
        return self._idx2feature[idx]
    
    def feature2idx(self, feature: str) -> int:
        """Returns the index for a given feature name."""
        return self._feature2idx[feature]
    
    def idx2lexunit(self, idx: int) -> str:
        """Returns the lex unit name for a given index."""
        return self._idx2lexunit[idx]
    
    def lexunit2idx(self, lexunit: str) -> int:
        """Returns the index for a given lex unit name."""
        return self._lexunit2idx[lexunit]
    
    def make_precision_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the precision calculation: occurrences of word 1 with feature 1 / occurrences of all words with feature 1"""
        self.precision_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        column_sums = co_occurrence_matrix.sum(axis=0)
        self.precision_matrix = co_occurrence_matrix / column_sums
        # Avoid division by zero
        self.precision_matrix[np.isnan(self.precision_matrix)] = 0
        self.feature_matrix = self.precision_matrix

    def make_coverage_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the coverage calculation: occurrences of word 1 with feature 1 / occurrences of word 1"""
        self.coverage_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        self.coverage_matrix = co_occurrence_matrix / row_sums[:, np.newaxis]
        # Avoid division by zero
        self.coverage_matrix[np.isnan(self.coverage_matrix)] = 0
        self.feature_matrix = self.coverage_matrix

    def make_geometric_mean_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the geometric mean calculation: sqrt(precision * coverage)"""
        self.geometric_mean_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        self.precision_matrix = co_occurrence_matrix / co_occurrence_matrix.sum(axis=0)
        self.coverage_matrix = co_occurrence_matrix / co_occurrence_matrix.sum(axis=1)[:, np.newaxis]
        # Avoid division by zero
        self.precision_matrix[np.isnan(self.precision_matrix)] = 0
        self.coverage_matrix[np.isnan(self.coverage_matrix)] = 0
        self.geometric_mean_matrix = np.sqrt(
            self.precision_matrix * self.coverage_matrix
        )
        # Avoid division by zero
        self.feature_matrix = self.geometric_mean_matrix

    def make_pmi_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the PMI calculation: log( w1 with feature 1 * sum of co-occurrence matrix / sum of line * sum of column)"""
        self.pmi_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        column_sums = co_occurrence_matrix.sum(axis=0)
        total_sum = co_occurrence_matrix.sum()
        for i in range(len(self._idx2lexunit)):
            for j in range(len(self._idx2feature)):
                if co_occurrence_matrix[i, j] > 0:
                    self.pmi_matrix[i, j] = np.log(
                        (co_occurrence_matrix[i, j] * total_sum) /
                        (row_sums[i] * column_sums[j])
                    )
                else:
                    self.pmi_matrix[i, j] = 0  # Avoid log(0)

        self.feature_matrix = self.pmi_matrix
    
    def make_tfidf_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the tf-idf calculation: tf = occurrences of word 1 with feature 1 / occurrences of all other features for word 1 and idf = log(total number of lexical units / the number of lexical units with feature 1)"""
        self.tfidf_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        total_lexunits = co_occurrence_matrix.shape[0]
        column_sums = co_occurrence_matrix.sum(axis=0)
        for i in range(len(self._idx2lexunit)):
            for j in range(len(self._idx2feature)):
                if co_occurrence_matrix[i, j] > 0:
                    tf = co_occurrence_matrix[i, j] / row_sums[i]
                    idf = np.log(total_lexunits / column_sums[j])
                    self.tfidf_matrix[i, j] = tf * idf
                else:
                    self.tfidf_matrix[i, j] = 0
        self.feature_matrix = self.tfidf_matrix
        
    def show_variance_graphs(self):
        self.make_coverage_matrix(self.co_occurrence_matrix)
        X_coverage = self.feature_matrix

        self.make_precision_matrix(self.co_occurrence_matrix)
        X_precision = self.feature_matrix

        self.make_pmi_matrix(self.co_occurrence_matrix)
        X_pmi = self.feature_matrix

        self.make_geometric_mean_matrix(self.co_occurrence_matrix)
        X_geometric_mean = self.feature_matrix

        self.make_tfidf_matrix(self.co_occurrence_matrix)
        X_tfidf = self.feature_matrix

        # Calculate variance for each column in the matrices
        variances = {
            "Coverage": np.var(X_coverage, axis=0),
            "Precision": np.var(X_precision, axis=0),
            "PMI": np.var(X_pmi, axis=0),
            "TF-IDF": np.var(X_tfidf, axis=0),
            "Geometric Mean": np.var(X_geometric_mean, axis=0),
        }

        # Create subplots
        fig = make_subplots(rows=5, cols=1, subplot_titles=list(variances.keys()), shared_xaxes=True)

        # Add bar plots for each matrix
        for i, (name, var) in enumerate(variances.items(), start=1):
            sorted_indices = np.argsort(var)  # Get indices that would sort the variance
            sorted_variances = var[sorted_indices]  # Sort variances
            sorted_features = [self.idx2feature(idx) for idx in sorted_indices]  # Map indices to features
            fig.add_trace(
                go.Bar(
                    x=list(range(len(sorted_variances))),
                    y=sorted_variances,
                    text=sorted_features,  # Feature names
                    hovertemplate="Feature: %{text}<br>Variance: %{y}<extra></extra>",
                ),
                row=i,
                col=1,
            )

        # Update layout
        fig.update_layout(
            height=1500,  # Adjust height for better visualization
            title="Variance per Column for Each Matrix",
            showlegend=False,
        )

        # Show plot
        fig.show()

        return X_coverage, X_precision, X_pmi, X_geometric_mean, X_tfidf
    
    def _filter_features(self, features, excluded_patterns):
        """Filter out features matching any of the excluded patterns."""
        import re
        filtered_features = []
        for feature in features:
            should_exclude = False
            for pattern in excluded_patterns:
                if re.search(pattern, feature):
                    should_exclude = True
                    break
            if not should_exclude:
                filtered_features.append(feature)
        return filtered_features

    def _row_vector(self, idx: int) -> np.ndarray:
        """Internal: returns the feature vector for a given row index."""
        return self.feature_matrix[idx]

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """1 - cosine similarity. Returns 1.0 if any vector is all-zero."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return 1.0
        return 1.0 - (np.dot(a, b) / (na * nb))

    def _euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def find_far_apart_lexunits(
        self,
        metric: str = "cosine",
        threshold: float | None = None,
        top_k: int = 50,
        only_cross_treebank: bool = True,
        print_results: bool = True,
        same_lemma: bool = False,
    ):
        """
        Compute distances between vectors of the same (lemma, POS) across treebanks
        and return the farthest pairs.

        Args:
            metric: "cosine" or "euclidean".
            threshold: if set, only return pairs with distance >= threshold.
            top_k: maximum number of pairs to return (after sorting by distance desc).
            only_cross_treebank: if True, ignore pairs from the same treebank.
            print_results: if True, print a compact report.

        Returns:
            List of dicts: {
                'lemma': str, 'pos': str,
                'treebank_a': str, 'treebank_b': str,
                'idx_a': int, 'idx_b': int,
                'distance': float
            }
        """
        # Group row indices by (lemma, pos)
        groups = {}

        for idx, lexunit in self._idx2lexunit.items():
  
            lemma, pos, treebank = lexunit

            if same_lemma:
                key = (lemma, pos)
            else:
                key = (pos)
            groups.setdefault(key, []).append(idx)


        # Distance selector
        if metric == "cosine":
            dist_fn = self._cosine_distance
        elif metric == "euclidean":
            dist_fn = self._euclidean_distance
        else:
            raise ValueError("Unsupported metric. Use 'cosine' or 'euclidean'.")

        results = []
        for group_key, idxs in groups.items():
            if len(idxs) < 2:
                continue
            for i, j in combinations(idxs, 2):
                lu_i = self._idx2lexunit[i]
                lu_j = self._idx2lexunit[j]

                lemma_i, pos_i, tb_i = lu_i
                lemma_j, pos_j, tb_j = lu_j

                if only_cross_treebank and tb_i == tb_j:
                    continue
                vi = self._row_vector(i)
                vj = self._row_vector(j)
                d = dist_fn(vi, vj)
                if threshold is None or d >= threshold:
                    results.append(
                        {
                            "lemma_a": lemma_i,
                            "lemma_b": lemma_j,
                            "pos": pos_i,
                            "treebank_a": tb_i,
                            "treebank_b": tb_j,
                            "idx_a": i,
                            "idx_b": j,
                            "distance": d,
                        }
                    )

        # Sort by distance desc and cap
        results.sort(key=lambda r: r["distance"], reverse=True)
        if top_k is not None:
            results = results[:top_k]
    
        if print_results:
            if not results:
                print(f"There are no lexical units farther apart than the given threshold ({threshold}) using metric={metric}.")
            else:
                print(f"Top {len(results)} farthest lexical unit pairs (metric={metric}):")
                print(f"{'PAIR':<30} {'TREEBANKS':<40} {'DISTANCE'}")
                print("-" * 80)
            
                for r in results:
                    # Format: "le vs il (DET)"
                    pair_str = f"{r['lemma_a']} vs {r['lemma_b']} ({r['pos']})"
                    tb_str = f"{r['treebank_a']} vs {r['treebank_b']}"
                    print(f"{pair_str:<30} {tb_str:<40} {r['distance']:.4f}")

        return results

class CorpusSyntacticFunctions:
    def __init__(
        self,
        treebank_path: str,
        grew_pattern: str,
        patterns_text_file: str,
        use_sud: bool = False,
        min_occurrences: int = 10,
        matrix_type: str = "PMI"
    ):
        if use_sud:
            grewpy.set_config("sud")
        else:
            grewpy.set_config("ud")

        corpus = grewpy.Corpus(treebank_path)
        draft = grewpy.CorpusDraft(corpus)

        # all_matches is a dict with keys as matching nodes and values as a list of dictionaries of this kind of structure: 'chat': [{'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}]
        # for me this is because i clustered by lemma, so i get lemmas as keys
        all_matches = corpus.search(
            grewpy.Request(grew_pattern)
            .without("X[InIdiom=Yes]")
            .without("X[Idiom=Yes]")
            .without("X[InTitle=Yes]")
            .without("X[Title=Yes]")
            .without("X[Scrap=Yes]")
            .without("X[Foreign]")
            .without("X[Lang]")
            .without("X-[fixed]->Y")
            .without("Y-[flat:name]->X")
            .without("Y-[goeswith]->X"),
            clustering_parameter=["e.label", "X.upos", "Y.upos"],
        ) # getting rid of noisy matches

        sent_id_to_sentence = {
            draft[i].meta["sent_id"]: draft[i].features for i in range(len(draft)) # type: ignore
        }

        match_dep = {}
        for dep, x_poses in all_matches.items():
            for x_pos, y_poses in x_poses.items():
                for y_pos, matches in y_poses.items():
                    for m in matches:
                        match_sent_id = m["sent_id"]
                        x_node_index = str(m["matching"]["nodes"]["X"])
                        y_node_index = str(m["matching"]["nodes"]["Y"])
                        key = (x_pos, dep, y_pos)
                        if match_sent_id in sent_id_to_sentence:
                                current_sentence_features = sent_id_to_sentence[match_sent_id]
                                if x_node_index in current_sentence_features.keys() and y_node_index in current_sentence_features.keys():
                                    current_x_token_features = current_sentence_features[
                                        x_node_index
                                    ]
                                    current_y_token_features = current_sentence_features[
                                        y_node_index
                                    ]
                                    if "ExtPos" in current_x_token_features:
                                        x_pos = current_x_token_features["ExtPos"]
                                        key = (x_pos, dep, y_pos)
                                    if "ExtPos" in current_y_token_features:
                                        y_pos = current_y_token_features["ExtPos"]
                                        key = (x_pos, dep, y_pos)
                                    
                                    if key not in match_dep:
                                        match_dep[key] = []
                                    if m not in match_dep[key]:    # to avoid duplicates, because i count every relation twice, once from the perspective of X and once from the perspective of Y
                                        match_dep[key].append(m)

        copy_match_dep = match_dep.copy()
        for k, v in copy_match_dep.items():
            if len(v) < min_occurrences:
                match_dep.pop(k)
        with open(patterns_text_file) as in_stream:
            config = yaml.load(in_stream, Loader=yaml.Loader)

        templates = grex.utils.FeaturePredicate.from_config(config["templates"])
        feature_predicate = grex.utils.FeaturePredicate.from_config(
            config["features"], templates=templates
        )
        
        data = {k: list() for k in match_dep}
        for dep_unit, mts in match_dep.items():
            for match in mts:
                features = grex.data.extract_features(draft, match, feature_predicate)
                formatted_features = [
                            (
                                f"{':'.join(k)}={v}"
                                if not isinstance(v, set) # if the value is a set
                                else f"{':'.join(k)}={val}" # if the value is a set, we want to iterate through it and create a string for each value
                            )
                            for k, v in features.items()
                            for val in (v if isinstance(v, set) else [v])
                ]

                for feat in formatted_features:
                    if re.findall("Cxn", feat):
                        formatted_features.remove(feat)

                data[dep_unit].append(formatted_features)

        unique_dep_units = sorted(set([k for k in data]))
        unique_features = sorted(
            set(
                [
                    feat
                    for _, match_dep in data.items() # iterating through (lemma, pos) : list of features for each match
                    for m in match_dep # iterating through each feature for the match
                    for feat in m
                ]
            )
        )

        self._idx2feature = {i: feat for i, feat in enumerate(unique_features)}
        self._feature2idx = {feat: i for i, feat in self._idx2feature.items()}
        self._idx2lexunit = {i: feat for i, feat in enumerate(unique_dep_units)}
        self._lexunit2idx = {feat: i for i, feat in self._idx2lexunit.items()}

        self.co_occurrence_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        for lexunit, feature_lists in data.items():
            for feature_list in feature_lists:
                for feature in feature_list:
                    self.co_occurrence_matrix[self._lexunit2idx[lexunit], self._feature2idx[feature]] += 1

        if matrix_type == "precision":
            self.make_precision_matrix(self.co_occurrence_matrix)
        elif matrix_type == "coverage":
            self.make_coverage_matrix(self.co_occurrence_matrix)
        elif matrix_type == "PMI":
            self.make_pmi_matrix(self.co_occurrence_matrix)
        elif matrix_type == "tf-idf":
            self.make_tfidf_matrix(self.co_occurrence_matrix)
        elif matrix_type == "geometric_mean":
            self.make_geometric_mean_matrix(self.co_occurrence_matrix)
        else:
            raise ValueError("Invalid matrix type. Choose from 'precision', 'coverage', 'PMI', or 'tf-idf'.")


    def idx2feature(self, idx: int) -> str:
        """Returns the feature name for a given index."""
        return self._idx2feature[idx]
    
    def feature2idx(self, feature: str) -> int:
        """Returns the index for a given feature name."""
        return self._feature2idx[feature]
    
    def idx2lexunit(self, idx: int) -> str:
        """Returns the lex unit name for a given index."""
        return self._idx2lexunit[idx]
    
    def lexunit2idx(self, lexunit: str) -> int:
        """Returns the index for a given lex unit name."""
        return self._lexunit2idx[lexunit]
    
    def make_precision_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the precision calculation: occurrences of word 1 with feature 1 / occurrences of all words with feature 1"""
        self.precision_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        column_sums = co_occurrence_matrix.sum(axis=0)
        self.precision_matrix = co_occurrence_matrix / column_sums
        # Avoid division by zero
        self.precision_matrix[np.isnan(self.precision_matrix)] = 0
        self.feature_matrix = self.precision_matrix

    def make_coverage_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the coverage calculation: occurrences of word 1 with feature 1 / occurrences of word 1"""
        self.coverage_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        self.coverage_matrix = co_occurrence_matrix / row_sums[:, np.newaxis]
        # Avoid division by zero
        self.coverage_matrix[np.isnan(self.coverage_matrix)] = 0
        self.feature_matrix = self.coverage_matrix

    def make_geometric_mean_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the geometric mean calculation: sqrt(precision * coverage)"""
        self.geometric_mean_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        self.precision_matrix = co_occurrence_matrix / co_occurrence_matrix.sum(axis=0)
        self.coverage_matrix = co_occurrence_matrix / co_occurrence_matrix.sum(axis=1)[:, np.newaxis]
        # Avoid division by zero
        self.precision_matrix[np.isnan(self.precision_matrix)] = 0
        self.coverage_matrix[np.isnan(self.coverage_matrix)] = 0
        self.geometric_mean_matrix = np.sqrt(
            self.precision_matrix * self.coverage_matrix
        )
        # Avoid division by zero
        self.feature_matrix = self.geometric_mean_matrix

    def make_pmi_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the PMI calculation: log( w1 with feature 1 * sum of co-occurrence matrix / sum of line * sum of column)"""
        self.pmi_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        column_sums = co_occurrence_matrix.sum(axis=0)
        total_sum = co_occurrence_matrix.sum()
        for i in range(len(self._idx2lexunit)):
            for j in range(len(self._idx2feature)):
                if co_occurrence_matrix[i, j] > 0:
                    self.pmi_matrix[i, j] = np.log(
                        (co_occurrence_matrix[i, j] * total_sum) /
                        (row_sums[i] * column_sums[j])
                    )
                else:
                    self.pmi_matrix[i, j] = 0  # Avoid log(0)

        self.feature_matrix = self.pmi_matrix
    
    def make_tfidf_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the tf-idf calculation: tf = occurrences of word 1 with feature 1 / occurrences of all other features for word 1 and idf = log(total number of lexical units / the number of lexical units with feature 1)"""
        self.tfidf_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        total_lexunits = co_occurrence_matrix.shape[0]
        column_sums = co_occurrence_matrix.sum(axis=0)
        for i in range(len(self._idx2lexunit)):
            for j in range(len(self._idx2feature)):
                if co_occurrence_matrix[i, j] > 0:
                    tf = co_occurrence_matrix[i, j] / row_sums[i]
                    idf = np.log(total_lexunits / column_sums[j])
                    self.tfidf_matrix[i, j] = tf * idf
                else:
                    self.tfidf_matrix[i, j] = 0
        self.feature_matrix = self.tfidf_matrix
        
    def show_variance_graphs(self):
        self.make_coverage_matrix(self.co_occurrence_matrix)
        X_coverage = self.feature_matrix

        self.make_precision_matrix(self.co_occurrence_matrix)
        X_precision = self.feature_matrix

        self.make_pmi_matrix(self.co_occurrence_matrix)
        X_pmi = self.feature_matrix

        self.make_geometric_mean_matrix(self.co_occurrence_matrix)
        X_geometric_mean = self.feature_matrix

        self.make_tfidf_matrix(self.co_occurrence_matrix)
        X_tfidf = self.feature_matrix

        # Calculate variance for each column in the matrices
        variances = {
            "Coverage": np.var(X_coverage, axis=0),
            "Precision": np.var(X_precision, axis=0),
            "PMI": np.var(X_pmi, axis=0),
            "TF-IDF": np.var(X_tfidf, axis=0),
            "Geometric Mean": np.var(X_geometric_mean, axis=0),
        }

        # Create subplots
        fig = make_subplots(rows=5, cols=1, subplot_titles=list(variances.keys()), shared_xaxes=True)

        # Add bar plots for each matrix
        for i, (name, var) in enumerate(variances.items(), start=1):
            sorted_indices = np.argsort(var)  # Get indices that would sort the variance
            sorted_variances = var[sorted_indices]  # Sort variances
            sorted_features = [self.idx2feature(idx) for idx in sorted_indices]  # Map indices to features
            fig.add_trace(
                go.Bar(
                    x=list(range(len(sorted_variances))),
                    y=sorted_variances,
                    text=sorted_features,  # Feature names
                    hovertemplate="Feature: %{text}<br>Variance: %{y}<extra></extra>",
                ),
                row=i,
                col=1,
            )

        # Update layout
        fig.update_layout(
            height=1500,  # Adjust height for better visualization
            title="Variance per Column for Each Matrix",
            showlegend=False,
        )

        # Show plot
        fig.show()

        return X_coverage, X_precision, X_pmi, X_geometric_mean, X_tfidf
    
class CorpusNoPos:
    def __init__(
        self,
        treebank_path: str,
        grew_pattern: str,
        patterns_text_file: str,
        use_sud: bool = False,
        min_occurrences: int = 10,
        matrix_type: str = "PMI"
    ):
        if use_sud:
            grewpy.set_config("sud")
        else:
            grewpy.set_config("ud")

        corpus = grewpy.Corpus(treebank_path)
        draft = grewpy.CorpusDraft(corpus)

        # all_matches is a dict with keys as matching nodes and values as a list of dictionaries of this kind of structure: 'chat': [{'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}]
        # for me this is because i clustered by lemma, so i get lemmas as keys
        all_matches = corpus.search(
            grewpy.Request(grew_pattern)
            .without("X[InIdiom=Yes]")
            .without("X[Idiom=Yes]")
            .without("X[InTitle=Yes]")
            .without("X[Title=Yes]")
            .without("X[Scrap=Yes]")
            .without("X[Foreign]")
            .without("X[Lang]")
            .without("X-[fixed]->Y")
            .without("Y-[flat:name]->X")
            .without("Y-[goeswith]->X"),
            clustering_parameter=["X.lemma"],
        ) # getting rid of noisy matches

        matches = {}
        for key, value in all_matches.items(): # type: ignore
            # remove those that have less than 10 occurrences
            if len(value) > min_occurrences:
                matches[key] = value

        # Create a dictionary to map sent_id to sentences for quick lookup
        # So for each sent_id I have each node and its features:  {'fr-ud-train_00001': {'0': {'form': '__0__'},'1': {'Definite': 'Def','Number': 'Plur', 'form': 'Les'}, '2': {'Gender': 'Fem','Number': 'Plur','form': 'commotions',}} etc
        sent_id_to_sentence = {
            draft[i].meta["sent_id"]: draft[i].features for i in range(len(draft)) # type: ignore
        }

        match_upos = {}
        # iterating through each lemma and its values like this: '€': [{'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}]
        for key, value in matches.items():
            # iterating through each match for the lemma so for example {'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}
            for m in value:
                match_sent_id = m["sent_id"]
                match_node_index = str(m["matching"]["nodes"]["X"])
                if match_sent_id in sent_id_to_sentence:
                    current_sentence_features = sent_id_to_sentence[match_sent_id]
                    if match_node_index in current_sentence_features.keys():
                        current_token_features = current_sentence_features[
                            match_node_index
                        ]
                        # if the token has a feature called ExtPos, we use that as the key, otherwise we use the upos feature
                        # ExtPos is used for example in 10% -> % has upos SYM but ExtPos Noun , so we want to use noun
                        # and what we're doing is creating a dictionary with keys (lemma, pos) and values as a list of matches
                        # so for example ('reason', 'NOUN') -> [match1, match2, match3] where a match looks like this: {'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}
                        if "ExtPos" in current_token_features:
                            match_upos.setdefault(
                                (key, current_token_features["ExtPos"]), []
                            ).append(m)
                        else:
                            match_upos.setdefault(
                                (key, current_token_features["upos"]), []
                            ).append(m)

        # remove those that have less than 10 occurrences again because now we've split by pos so we might have some that are less than 10 
        # for example if we had reason with 11 occurrences before and now we have reason:VERB with 5 occurrences and reason:NOUN with 6 occurrences
        new_match_upos = {}
        for key, value in match_upos.items():
            if len(value) > min_occurrences:
                new_match_upos[key] = value
        match_upos = new_match_upos

        nb_matches = 0
        for key, value in match_upos.items():
            nb_matches += len(value)
        print(f"Number of matches after filtering: {nb_matches}")

        # grex stuff
        with open(patterns_text_file) as in_stream:
            config = yaml.load(in_stream, Loader=yaml.Loader)

        templates = grex.utils.FeaturePredicate.from_config(config["templates"])
        feature_predicate = grex.utils.FeaturePredicate.from_config(
            config["features"], templates=templates
        )

        # data is a dict where each (lemma, pos) is a key and the value is an empty list for now
        data = {k: list() for k in match_upos}
        for lex_unit, mts in match_upos.items():
            for match in mts:
                features = grex.data.extract_features(draft, match, feature_predicate)
                # features looks like this: features={('node', 'X', 'own', 'Number'): 'Plur', ('node', 'X', 'own', 'rel_shallow'): 'comp:obj'
                keys_to_delete = []
                for f, feature_value in features.items():
                    if f[3] == 'upos':
                        keys_to_delete.append(f)
                
                for k in keys_to_delete:
                    features.pop(k)
                    
                formatted_features = [
                    (
                        f"{':'.join(k)}={v}"
                        if not isinstance(v, set) # if the value is a set
                        else f"{':'.join(k)}={val}" # if the value is a set, we want to iterate through it and create a string for each value
                    )
                    for k, v in features.items()
                    for val in (v if isinstance(v, set) else [v])
                ]
                # so what we did is go from features = {('key1', 'key2'): 'value1', ('key3',): {'value2', 'value3'}} to ['key1:key2=value1', 'key3=value2', 'key3=value3']
                # and now we append the features to the data dict for the corresponding (lemma, pos) key
                data[lex_unit].append(formatted_features)

        unique_lemma = sorted(set([k for k in data]))
        unique_features = sorted(
            set(
                [
                    feat
                    for _, match_upos in data.items() # iterating through (lemma, pos) : list of features for each match
                    for m in match_upos # iterating through each feature for the match
                    for feat in m
                ]
            )
        )

        self._idx2feature = {i: feat for i, feat in enumerate(unique_features)}
        self._feature2idx = {feat: i for i, feat in self._idx2feature.items()}
        self._idx2lexunit = {i: feat for i, feat in enumerate(unique_lemma)}
        self._lexunit2idx = {feat: i for i, feat in self._idx2lexunit.items()}

        self.co_occurrence_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        for lexunit, feature_lists in data.items():
            for feature_list in feature_lists:
                for feature in feature_list:
                    self.co_occurrence_matrix[self._lexunit2idx[lexunit], self._feature2idx[feature]] += 1

        if matrix_type == "precision":
            self.make_precision_matrix(self.co_occurrence_matrix)
        elif matrix_type == "coverage":
            self.make_coverage_matrix(self.co_occurrence_matrix)
        elif matrix_type == "PMI":
            self.make_pmi_matrix(self.co_occurrence_matrix)
        elif matrix_type == "tf-idf":
            self.make_tfidf_matrix(self.co_occurrence_matrix)
        elif matrix_type == "geometric_mean":
            self.make_geometric_mean_matrix(self.co_occurrence_matrix)
        else:
            raise ValueError("Invalid matrix type. Choose from 'precision', 'coverage', 'PMI', or 'tf-idf'.")


    def idx2feature(self, idx: int) -> str:
        """Returns the feature name for a given index."""
        return self._idx2feature[idx]
    
    def feature2idx(self, feature: str) -> int:
        """Returns the index for a given feature name."""
        return self._feature2idx[feature]
    
    def idx2lexunit(self, idx: int) -> str:
        """Returns the lex unit name for a given index."""
        return self._idx2lexunit[idx]
    
    def lexunit2idx(self, lexunit: str) -> int:
        """Returns the index for a given lex unit name."""
        return self._lexunit2idx[lexunit]
    
    def make_precision_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the precision calculation: occurrences of word 1 with feature 1 / occurrences of all words with feature 1"""
        self.precision_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        column_sums = co_occurrence_matrix.sum(axis=0)
        self.precision_matrix = co_occurrence_matrix / column_sums
        # Avoid division by zero
        self.precision_matrix[np.isnan(self.precision_matrix)] = 0
        self.feature_matrix = self.precision_matrix

    def make_coverage_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the coverage calculation: occurrences of word 1 with feature 1 / occurrences of word 1"""
        self.coverage_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        self.coverage_matrix = co_occurrence_matrix / row_sums[:, np.newaxis]
        # Avoid division by zero
        self.coverage_matrix[np.isnan(self.coverage_matrix)] = 0
        self.feature_matrix = self.coverage_matrix

    def make_geometric_mean_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the geometric mean calculation: sqrt(precision * coverage)"""
        self.geometric_mean_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        self.precision_matrix = co_occurrence_matrix / co_occurrence_matrix.sum(axis=0)
        self.coverage_matrix = co_occurrence_matrix / co_occurrence_matrix.sum(axis=1)[:, np.newaxis]
        # Avoid division by zero
        self.precision_matrix[np.isnan(self.precision_matrix)] = 0
        self.coverage_matrix[np.isnan(self.coverage_matrix)] = 0
        self.geometric_mean_matrix = np.sqrt(
            self.precision_matrix * self.coverage_matrix
        )
        # Avoid division by zero
        self.feature_matrix = self.geometric_mean_matrix

    def make_pmi_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the PMI calculation: log( w1 with feature 1 * sum of co-occurrence matrix / sum of line * sum of column)"""
        self.pmi_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        column_sums = co_occurrence_matrix.sum(axis=0)
        total_sum = co_occurrence_matrix.sum()
        for i in range(len(self._idx2lexunit)):
            for j in range(len(self._idx2feature)):
                if co_occurrence_matrix[i, j] > 0:
                    self.pmi_matrix[i, j] = np.log(
                        (co_occurrence_matrix[i, j] * total_sum) /
                        (row_sums[i] * column_sums[j])
                    )
                else:
                    self.pmi_matrix[i, j] = 0  # Avoid log(0)

        self.feature_matrix = self.pmi_matrix
    
    def make_tfidf_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the tf-idf calculation: tf = occurrences of word 1 with feature 1 / occurrences of all other features for word 1 and idf = log(total number of lexical units / the number of lexical units with feature 1)"""
        self.tfidf_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        total_lexunits = co_occurrence_matrix.shape[0]
        column_sums = co_occurrence_matrix.sum(axis=0)
        for i in range(len(self._idx2lexunit)):
            for j in range(len(self._idx2feature)):
                if co_occurrence_matrix[i, j] > 0:
                    tf = co_occurrence_matrix[i, j] / row_sums[i]
                    idf = np.log(total_lexunits / column_sums[j])
                    self.tfidf_matrix[i, j] = tf * idf
                else:
                    self.tfidf_matrix[i, j] = 0
        self.feature_matrix = self.tfidf_matrix
        
    def show_variance_graphs(self):
        self.make_coverage_matrix(self.co_occurrence_matrix)
        X_coverage = self.feature_matrix

        self.make_precision_matrix(self.co_occurrence_matrix)
        X_precision = self.feature_matrix

        self.make_pmi_matrix(self.co_occurrence_matrix)
        X_pmi = self.feature_matrix

        self.make_geometric_mean_matrix(self.co_occurrence_matrix)
        X_geometric_mean = self.feature_matrix

        self.make_tfidf_matrix(self.co_occurrence_matrix)
        X_tfidf = self.feature_matrix

        # Calculate variance for each column in the matrices
        variances = {
            "Coverage": np.var(X_coverage, axis=0),
            "Precision": np.var(X_precision, axis=0),
            "PMI": np.var(X_pmi, axis=0),
            "TF-IDF": np.var(X_tfidf, axis=0),
            "Geometric Mean": np.var(X_geometric_mean, axis=0),
        }

        # Create subplots
        fig = make_subplots(rows=5, cols=1, subplot_titles=list(variances.keys()), shared_xaxes=True)

        # Add bar plots for each matrix
        for i, (name, var) in enumerate(variances.items(), start=1):
            sorted_indices = np.argsort(var)  # Get indices that would sort the variance
            sorted_variances = var[sorted_indices]  # Sort variances
            sorted_features = [self.idx2feature(idx) for idx in sorted_indices]  # Map indices to features
            fig.add_trace(
                go.Bar(
                    x=list(range(len(sorted_variances))),
                    y=sorted_variances,
                    text=sorted_features,  # Feature names
                    hovertemplate="Feature: %{text}<br>Variance: %{y}<extra></extra>",
                ),
                row=i,
                col=1,
            )

        # Update layout
        fig.update_layout(
            height=1500,  # Adjust height for better visualization
            title="Variance per Column for Each Matrix",
            showlegend=False,
        )

        # Show plot
        fig.show()

        return X_coverage, X_precision, X_pmi, X_geometric_mean, X_tfidf
class IterativeCorpusNoPos:
    def __init__(
        self,
        clusters: dict,
        treebank_path: str,
        grew_pattern: str,
        patterns_text_file: str,
        use_sud: bool = False,
        min_occurrences: int = 10,
        matrix_type: str = "PMI"
    ):
        if use_sud:
            grewpy.set_config("sud")
        else:
            grewpy.set_config("ud")

        corpus = grewpy.Corpus(treebank_path)
        draft = grewpy.CorpusDraft(corpus)

        # all_matches is a dict with keys as matching nodes and values as a list of dictionaries of this kind of structure: 'chat': [{'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}]
        # for me this is because i clustered by lemma, so i get lemmas as keys
        all_matches = corpus.search(
            grewpy.Request(grew_pattern)
            .without("X[InIdiom=Yes]")
            .without("X[Idiom=Yes]")
            .without("X[InTitle=Yes]")
            .without("X[Title=Yes]")
            .without("X[Scrap=Yes]")
            .without("X[Foreign]")
            .without("X[Lang]")
            .without("X-[fixed]->Y")
            .without("Y-[flat:name]->X")
            .without("Y-[goeswith]->X"),
            clustering_parameter=["X.lemma"],
        ) # getting rid of noisy matches

        matches = {}
        for key, value in all_matches.items(): # type: ignore
            # remove those that have less than 10 occurrences
            if len(value) > min_occurrences:
                matches[key] = value

        # Create a dictionary to map sent_id to sentences for quick lookup
        # So for each sent_id I have each node and its features:  {'fr-ud-train_00001': {'0': {'form': '__0__'},'1': {'Definite': 'Def','Number': 'Plur', 'form': 'Les'}, '2': {'Gender': 'Fem','Number': 'Plur','form': 'commotions',}} etc
        sent_id_to_sentence = {
            draft[i].meta["sent_id"]: draft[i].features for i in range(len(draft)) # type: ignore
        }

        match_upos = {}
        # iterating through each lemma and its values like this: '€': [{'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}]
        for key, value in matches.items():
            # iterating through each match for the lemma so for example {'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}
            for m in value:
                match_sent_id = m["sent_id"]
                match_node_index = str(m["matching"]["nodes"]["X"])
                if match_sent_id in sent_id_to_sentence:
                    current_sentence_features = sent_id_to_sentence[match_sent_id]
                    if match_node_index in current_sentence_features.keys():
                        current_token_features = current_sentence_features[
                            match_node_index
                        ]
                        # if the token has a feature called ExtPos, we use that as the key, otherwise we use the upos feature
                        # ExtPos is used for example in 10% -> % has upos SYM but ExtPos Noun , so we want to use noun
                        # and what we're doing is creating a dictionary with keys (lemma, pos) and values as a list of matches
                        # so for example ('reason', 'NOUN') -> [match1, match2, match3] where a match looks like this: {'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}
                        if "ExtPos" in current_token_features:
                            match_upos.setdefault(
                                (key, current_token_features["ExtPos"]), []
                            ).append(m)
                        else:
                            match_upos.setdefault(
                                (key, current_token_features["upos"]), []
                            ).append(m)

        # remove those that have less than 10 occurrences again because now we've split by pos so we might have some that are less than 10 
        # for example if we had reason with 11 occurrences before and now we have reason:VERB with 5 occurrences and reason:NOUN with 6 occurrences
        new_match_upos = {}
        for key, value in match_upos.items():
            if len(value) > min_occurrences:
                new_match_upos[key] = value
        match_upos = new_match_upos

        nb_matches = 0
        for key, value in match_upos.items():
            nb_matches += len(value)
        print(f"Number of matches after filtering: {nb_matches}")

        # grex stuff
        with open(patterns_text_file) as in_stream:
            config = yaml.load(in_stream, Loader=yaml.Loader)

        templates = grex.utils.FeaturePredicate.from_config(config["templates"])
        feature_predicate = grex.utils.FeaturePredicate.from_config(
            config["features"], templates=templates
        )

        # data is a dict where each (lemma, pos) is a key and the value is an empty list for now
        data = {k: list() for k in match_upos}
        not_in_clusters = set()
        for lex_unit, mts in match_upos.items():
            for match in mts:
                features = grex.data.extract_features(draft, match, feature_predicate)
                # features looks like this: features={('node', 'X', 'own', 'Number'): 'Plur', ('node', 'X', 'own', 'rel_shallow'): 'comp:obj'
                cluster_dict = {}
                keys_to_delete = []
                for f, feature_value in features.items():
                    if f[3] == "lemma":
                        keys_to_delete.append(f)
                        if type(feature_value) == set:
                            feature_values = list(feature_value)
                            cluster_dict[tuple((f[0], f[1], f[2], "cluster"))] = set()
                            for fv in feature_values:
                                if fv in clusters:
                                    cluster_dict[tuple((f[0], f[1], f[2], "cluster"))].add(clusters[fv])
                                else:
                                    not_in_clusters.add(fv)
                                    
                        else:
                            if feature_value in clusters:
                                cluster_dict[tuple((f[0], f[1], f[2], "cluster"))] = clusters[feature_value]
                            else:
                                not_in_clusters.add(feature_value)
                    elif f[3] == "upos":
                        keys_to_delete.append(f)

                features.update(cluster_dict)
                for k in keys_to_delete:
                    features.pop(k)
  
        # # print(f"Not in clusters: {not_in_clusters}")
        # # print("---")
        # # print(features)
                
                formatted_features = [
                    (
                        f"{':'.join(k)}={v}"
                        if not isinstance(v, set) # if the value is a set
                        else f"{':'.join(k)}={val}" # if the value is a set, we want to iterate through it and create a string for each value
                    )
                    for k, v in features.items()
                    for val in (v if isinstance(v, set) else [v])
                ]
                # so what we did is go from features = {('key1', 'key2'): 'value1', ('key3',): {'value2', 'value3'}} to ['key1:key2=value1', 'key3=value2', 'key3=value3']
                # and now we append the features to the data dict for the corresponding (lemma, pos) key
                data[lex_unit].append(formatted_features)

        unique_lemma = sorted(set([k for k in data]))
        unique_features = sorted(
            set(
                [
                    feat
                    for _, match_upos in data.items() # iterating through (lemma, pos) : list of features for each match
                    for m in match_upos # iterating through each feature for the match
                    for feat in m
                ]
            )
        )
        # print(unique_features)

        self._idx2feature = {i: feat for i, feat in enumerate(unique_features)}
        self._feature2idx = {feat: i for i, feat in self._idx2feature.items()}
        self._idx2lexunit = {i: feat for i, feat in enumerate(unique_lemma)}
        self._lexunit2idx = {feat: i for i, feat in self._idx2lexunit.items()}

        self.co_occurrence_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        for lexunit, feature_lists in data.items():
            for feature_list in feature_lists:
                for feature in feature_list:
                    self.co_occurrence_matrix[self._lexunit2idx[lexunit], self._feature2idx[feature]] += 1

        if matrix_type == "precision":
            self.make_precision_matrix(self.co_occurrence_matrix)
        elif matrix_type == "coverage":
            self.make_coverage_matrix(self.co_occurrence_matrix)
        elif matrix_type == "PMI":
            self.make_pmi_matrix(self.co_occurrence_matrix)
        elif matrix_type == "tf-idf":
            self.make_tfidf_matrix(self.co_occurrence_matrix)
        elif matrix_type == "geometric_mean":
            self.make_geometric_mean_matrix(self.co_occurrence_matrix)
        else:
            raise ValueError("Invalid matrix type. Choose from 'precision', 'coverage', 'PMI', or 'tf-idf'.")


    def idx2feature(self, idx: int) -> str:
        """Returns the feature name for a given index."""
        return self._idx2feature[idx]
    
    def feature2idx(self, feature: str) -> int:
        """Returns the index for a given feature name."""
        return self._feature2idx[feature]
    
    def idx2lexunit(self, idx: int) -> str:
        """Returns the lex unit name for a given index."""
        return self._idx2lexunit[idx]
    
    def lexunit2idx(self, lexunit: str) -> int:
        """Returns the index for a given lex unit name."""
        return self._lexunit2idx[lexunit]
    
    def make_precision_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the precision calculation: occurrences of word 1 with feature 1 / occurrences of all words with feature 1"""
        self.precision_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        column_sums = co_occurrence_matrix.sum(axis=0)
        self.precision_matrix = co_occurrence_matrix / column_sums
        # Avoid division by zero
        self.precision_matrix[np.isnan(self.precision_matrix)] = 0
        self.feature_matrix = self.precision_matrix

    def make_coverage_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the coverage calculation: occurrences of word 1 with feature 1 / occurrences of word 1"""
        self.coverage_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        self.coverage_matrix = co_occurrence_matrix / row_sums[:, np.newaxis]
        # Avoid division by zero
        self.coverage_matrix[np.isnan(self.coverage_matrix)] = 0
        self.feature_matrix = self.coverage_matrix

    def make_geometric_mean_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the geometric mean calculation: sqrt(precision * coverage)"""
        self.geometric_mean_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        self.precision_matrix = co_occurrence_matrix / co_occurrence_matrix.sum(axis=0)
        self.coverage_matrix = co_occurrence_matrix / co_occurrence_matrix.sum(axis=1)[:, np.newaxis]
        # Avoid division by zero
        self.precision_matrix[np.isnan(self.precision_matrix)] = 0
        self.coverage_matrix[np.isnan(self.coverage_matrix)] = 0
        self.geometric_mean_matrix = np.sqrt(
            self.precision_matrix * self.coverage_matrix
        )
        # Avoid division by zero
        self.feature_matrix = self.geometric_mean_matrix

    def make_pmi_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the PMI calculation: log( w1 with feature 1 * sum of co-occurrence matrix / sum of line * sum of column)"""
        self.pmi_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        column_sums = co_occurrence_matrix.sum(axis=0)
        total_sum = co_occurrence_matrix.sum()
        for i in range(len(self._idx2lexunit)):
            for j in range(len(self._idx2feature)):
                if co_occurrence_matrix[i, j] > 0:
                    self.pmi_matrix[i, j] = np.log(
                        (co_occurrence_matrix[i, j] * total_sum) /
                        (row_sums[i] * column_sums[j])
                    )
                else:
                    self.pmi_matrix[i, j] = 0  # Avoid log(0)

        self.feature_matrix = self.pmi_matrix
    
    def make_tfidf_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the tf-idf calculation: tf = occurrences of word 1 with feature 1 / occurrences of all other features for word 1 and idf = log(total number of lexical units / the number of lexical units with feature 1)"""
        self.tfidf_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        total_lexunits = co_occurrence_matrix.shape[0]
        column_sums = co_occurrence_matrix.sum(axis=0)
        for i in range(len(self._idx2lexunit)):
            for j in range(len(self._idx2feature)):
                if co_occurrence_matrix[i, j] > 0:
                    tf = co_occurrence_matrix[i, j] / row_sums[i]
                    idf = np.log(total_lexunits / column_sums[j])
                    self.tfidf_matrix[i, j] = tf * idf
                else:
                    self.tfidf_matrix[i, j] = 0
        self.feature_matrix = self.tfidf_matrix
        
    def show_variance_graphs(self):
        self.make_coverage_matrix(self.co_occurrence_matrix)
        X_coverage = self.feature_matrix

        self.make_precision_matrix(self.co_occurrence_matrix)
        X_precision = self.feature_matrix

        self.make_pmi_matrix(self.co_occurrence_matrix)
        X_pmi = self.feature_matrix

        self.make_geometric_mean_matrix(self.co_occurrence_matrix)
        X_geometric_mean = self.feature_matrix

        self.make_tfidf_matrix(self.co_occurrence_matrix)
        X_tfidf = self.feature_matrix

        # Calculate variance for each column in the matrices
        variances = {
            "Coverage": np.var(X_coverage, axis=0),
            "Precision": np.var(X_precision, axis=0),
            "PMI": np.var(X_pmi, axis=0),
            "TF-IDF": np.var(X_tfidf, axis=0),
            "Geometric Mean": np.var(X_geometric_mean, axis=0),
        }

        # Create subplots
        fig = make_subplots(rows=5, cols=1, subplot_titles=list(variances.keys()), shared_xaxes=True)

        # Add bar plots for each matrix
        for i, (name, var) in enumerate(variances.items(), start=1):
            sorted_indices = np.argsort(var)  # Get indices that would sort the variance
            sorted_variances = var[sorted_indices]  # Sort variances
            sorted_features = [self.idx2feature(idx) for idx in sorted_indices]  # Map indices to features
            fig.add_trace(
                go.Bar(
                    x=list(range(len(sorted_variances))),
                    y=sorted_variances,
                    text=sorted_features,  # Feature names
                    hovertemplate="Feature: %{text}<br>Variance: %{y}<extra></extra>",
                ),
                row=i,
                col=1,
            )

        # Update layout
        fig.update_layout(
            height=1500,  # Adjust height for better visualization
            title="Variance per Column for Each Matrix",
            showlegend=False,
        )

        # Show plot
        fig.show()

        return X_coverage, X_precision, X_pmi, X_geometric_mean, X_tfidf

class IterativeCorpus:
    def __init__(
        self,
        clusters: dict,
        treebank_path: str,
        grew_pattern: str,
        patterns_text_file: str,
        use_sud: bool = False,
        min_occurrences: int = 10,
        matrix_type: str = "PMI"
    ):
        if use_sud:
            grewpy.set_config("sud")
        else:
            grewpy.set_config("ud")

        corpus = grewpy.Corpus(treebank_path)
        draft = grewpy.CorpusDraft(corpus)

        # all_matches is a dict with keys as matching nodes and values as a list of dictionaries of this kind of structure: 'chat': [{'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}]
        # for me this is because i clustered by lemma, so i get lemmas as keys
        all_matches = corpus.search(
            grewpy.Request(grew_pattern)
            .without("X[InIdiom=Yes]")
            .without("X[Idiom=Yes]")
            .without("X[InTitle=Yes]")
            .without("X[Title=Yes]")
            .without("X[Scrap=Yes]")
            .without("X[Foreign]")
            .without("X[Lang]")
            .without("X-[fixed]->Y")
            .without("Y-[flat:name]->X")
            .without("Y-[goeswith]->X"),
            clustering_parameter=["X.lemma"],
        ) # getting rid of noisy matches

        matches = {}
        for key, value in all_matches.items(): # type: ignore
            # remove those that have less than 10 occurrences
            if len(value) > min_occurrences:
                matches[key] = value

        # Create a dictionary to map sent_id to sentences for quick lookup
        # So for each sent_id I have each node and its features:  {'fr-ud-train_00001': {'0': {'form': '__0__'},'1': {'Definite': 'Def','Number': 'Plur', 'form': 'Les'}, '2': {'Gender': 'Fem','Number': 'Plur','form': 'commotions',}} etc
        sent_id_to_sentence = {
            draft[i].meta["sent_id"]: draft[i].features for i in range(len(draft)) # type: ignore
        }

        match_upos = {}
        # iterating through each lemma and its values like this: '€': [{'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}]
        for key, value in matches.items():
            # iterating through each match for the lemma so for example {'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}
            for m in value:
                match_sent_id = m["sent_id"]
                match_node_index = str(m["matching"]["nodes"]["X"])
                if match_sent_id in sent_id_to_sentence:
                    current_sentence_features = sent_id_to_sentence[match_sent_id]
                    if match_node_index in current_sentence_features.keys():
                        current_token_features = current_sentence_features[
                            match_node_index
                        ]
                        # if the token has a feature called ExtPos, we use that as the key, otherwise we use the upos feature
                        # ExtPos is used for example in 10% -> % has upos SYM but ExtPos Noun , so we want to use noun
                        # and what we're doing is creating a dictionary with keys (lemma, pos) and values as a list of matches
                        # so for example ('reason', 'NOUN') -> [match1, match2, match3] where a match looks like this: {'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}
                        if "ExtPos" in current_token_features:
                            match_upos.setdefault(
                                (key, current_token_features["ExtPos"]), []
                            ).append(m)
                        else:
                            match_upos.setdefault(
                                (key, current_token_features["upos"]), []
                            ).append(m)

        # remove those that have less than 10 occurrences again because now we've split by pos so we might have some that are less than 10 
        # for example if we had reason with 11 occurrences before and now we have reason:VERB with 5 occurrences and reason:NOUN with 6 occurrences
        new_match_upos = {}
        for key, value in match_upos.items():
            if len(value) > min_occurrences:
                new_match_upos[key] = value
        match_upos = new_match_upos

        nb_matches = 0
        for key, value in match_upos.items():
            nb_matches += len(value)
        print(f"Number of matches after filtering: {nb_matches}")

        # grex stuff
        with open(patterns_text_file) as in_stream:
            config = yaml.load(in_stream, Loader=yaml.Loader)

        templates = grex.utils.FeaturePredicate.from_config(config["templates"])
        feature_predicate = grex.utils.FeaturePredicate.from_config(
            config["features"], templates=templates
        )

        # data is a dict where each (lemma, pos) is a key and the value is an empty list for now
        data = {k: list() for k in match_upos}
        not_in_clusters = set()
        for lex_unit, mts in match_upos.items():
            for match in mts:
                features = grex.data.extract_features(draft, match, feature_predicate)
                # features looks like this: features={('node', 'X', 'own', 'Number'): 'Plur', ('node', 'X', 'own', 'rel_shallow'): 'comp:obj'
                cluster_dict = {}
                keys_to_delete = []
                for f, feature_value in features.items():
                    if f[3] == "lemma":
                        keys_to_delete.append(f)
                        if type(feature_value) == set:
                            feature_values = list(feature_value)
                            cluster_dict[tuple((f[0], f[1], f[2], "cluster"))] = set()
                            for fv in feature_values:
                                if fv in clusters:
                                    cluster_dict[tuple((f[0], f[1], f[2], "cluster"))].add(clusters[fv])
                                else:
                                    not_in_clusters.add(fv)
                                    
                        else:
                            if feature_value in clusters:
                                cluster_dict[tuple((f[0], f[1], f[2], "cluster"))] = clusters[feature_value]
                            else:
                                not_in_clusters.add(feature_value)
                        

                features.update(cluster_dict)
        # print(f"Not in clusters: {not_in_clusters}")
        # print("---")
        # print(features)
                for k in keys_to_delete:
                    features.pop(k)
              
                formatted_features = [
                    (
                        f"{':'.join(k)}={v}"
                        if not isinstance(v, set) # if the value is a set
                        else f"{':'.join(k)}={val}" # if the value is a set, we want to iterate through it and create a string for each value
                    )
                    for k, v in features.items()
                    for val in (v if isinstance(v, set) else [v])
                ]
                # so what we did is go from features = {('key1', 'key2'): 'value1', ('key3',): {'value2', 'value3'}} to ['key1:key2=value1', 'key3=value2', 'key3=value3']
                # and now we append the features to the data dict for the corresponding (lemma, pos) key
                data[lex_unit].append(formatted_features)

        unique_lemma = sorted(set([k for k in data]))
        unique_features = sorted(
            set(
                [
                    feat
                    for _, match_upos in data.items() # iterating through (lemma, pos) : list of features for each match
                    for m in match_upos # iterating through each feature for the match
                    for feat in m
                ]
            )
        )
        # print(unique_features)

        self._idx2feature = {i: feat for i, feat in enumerate(unique_features)}
        self._feature2idx = {feat: i for i, feat in self._idx2feature.items()}
        self._idx2lexunit = {i: feat for i, feat in enumerate(unique_lemma)}
        self._lexunit2idx = {feat: i for i, feat in self._idx2lexunit.items()}

        self.co_occurrence_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        for lexunit, feature_lists in data.items():
            for feature_list in feature_lists:
                for feature in feature_list:
                    self.co_occurrence_matrix[self._lexunit2idx[lexunit], self._feature2idx[feature]] += 1

        if matrix_type == "precision":
            self.make_precision_matrix(self.co_occurrence_matrix)
        elif matrix_type == "coverage":
            self.make_coverage_matrix(self.co_occurrence_matrix)
        elif matrix_type == "PMI":
            self.make_pmi_matrix(self.co_occurrence_matrix)
        elif matrix_type == "tf-idf":
            self.make_tfidf_matrix(self.co_occurrence_matrix)
        elif matrix_type == "geometric_mean":
            self.make_geometric_mean_matrix(self.co_occurrence_matrix)
        else:
            raise ValueError("Invalid matrix type. Choose from 'precision', 'coverage', 'PMI', or 'tf-idf'.")


    def idx2feature(self, idx: int) -> str:
        """Returns the feature name for a given index."""
        return self._idx2feature[idx]
    
    def feature2idx(self, feature: str) -> int:
        """Returns the index for a given feature name."""
        return self._feature2idx[feature]
    
    def idx2lexunit(self, idx: int) -> str:
        """Returns the lex unit name for a given index."""
        return self._idx2lexunit[idx]
    
    def lexunit2idx(self, lexunit: str) -> int:
        """Returns the index for a given lex unit name."""
        return self._lexunit2idx[lexunit]
    
    def make_precision_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the precision calculation: occurrences of word 1 with feature 1 / occurrences of all words with feature 1"""
        self.precision_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        column_sums = co_occurrence_matrix.sum(axis=0)
        self.precision_matrix = co_occurrence_matrix / column_sums
        # Avoid division by zero
        self.precision_matrix[np.isnan(self.precision_matrix)] = 0
        self.feature_matrix = self.precision_matrix

    def make_coverage_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the coverage calculation: occurrences of word 1 with feature 1 / occurrences of word 1"""
        self.coverage_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        self.coverage_matrix = co_occurrence_matrix / row_sums[:, np.newaxis]
        # Avoid division by zero
        self.coverage_matrix[np.isnan(self.coverage_matrix)] = 0
        self.feature_matrix = self.coverage_matrix

    def make_geometric_mean_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the geometric mean calculation: sqrt(precision * coverage)"""
        self.geometric_mean_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        self.precision_matrix = co_occurrence_matrix / co_occurrence_matrix.sum(axis=0)
        self.coverage_matrix = co_occurrence_matrix / co_occurrence_matrix.sum(axis=1)[:, np.newaxis]
        # Avoid division by zero
        self.precision_matrix[np.isnan(self.precision_matrix)] = 0
        self.coverage_matrix[np.isnan(self.coverage_matrix)] = 0
        self.geometric_mean_matrix = np.sqrt(
            self.precision_matrix * self.coverage_matrix
        )
        # Avoid division by zero
        self.feature_matrix = self.geometric_mean_matrix

    def make_pmi_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the PMI calculation: log( w1 with feature 1 * sum of co-occurrence matrix / sum of line * sum of column)"""
        self.pmi_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        column_sums = co_occurrence_matrix.sum(axis=0)
        total_sum = co_occurrence_matrix.sum()
        for i in range(len(self._idx2lexunit)):
            for j in range(len(self._idx2feature)):
                if co_occurrence_matrix[i, j] > 0:
                    self.pmi_matrix[i, j] = np.log(
                        (co_occurrence_matrix[i, j] * total_sum) /
                        (row_sums[i] * column_sums[j])
                    )
                else:
                    self.pmi_matrix[i, j] = 0  # Avoid log(0)

        self.feature_matrix = self.pmi_matrix
    
    def make_tfidf_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the tf-idf calculation: tf = occurrences of word 1 with feature 1 / occurrences of all other features for word 1 and idf = log(total number of lexical units / the number of lexical units with feature 1)"""
        self.tfidf_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        total_lexunits = co_occurrence_matrix.shape[0]
        column_sums = co_occurrence_matrix.sum(axis=0)
        for i in range(len(self._idx2lexunit)):
            for j in range(len(self._idx2feature)):
                if co_occurrence_matrix[i, j] > 0:
                    tf = co_occurrence_matrix[i, j] / row_sums[i]
                    idf = np.log(total_lexunits / column_sums[j])
                    self.tfidf_matrix[i, j] = tf * idf
                else:
                    self.tfidf_matrix[i, j] = 0
        self.feature_matrix = self.tfidf_matrix
        
    def show_variance_graphs(self):
        self.make_coverage_matrix(self.co_occurrence_matrix)
        X_coverage = self.feature_matrix

        self.make_precision_matrix(self.co_occurrence_matrix)
        X_precision = self.feature_matrix

        self.make_pmi_matrix(self.co_occurrence_matrix)
        X_pmi = self.feature_matrix

        self.make_geometric_mean_matrix(self.co_occurrence_matrix)
        X_geometric_mean = self.feature_matrix

        self.make_tfidf_matrix(self.co_occurrence_matrix)
        X_tfidf = self.feature_matrix

        # Calculate variance for each column in the matrices
        variances = {
            "Coverage": np.var(X_coverage, axis=0),
            "Precision": np.var(X_precision, axis=0),
            "PMI": np.var(X_pmi, axis=0),
            "TF-IDF": np.var(X_tfidf, axis=0),
            "Geometric Mean": np.var(X_geometric_mean, axis=0),
        }

        # Create subplots
        fig = make_subplots(rows=5, cols=1, subplot_titles=list(variances.keys()), shared_xaxes=True)

        # Add bar plots for each matrix
        for i, (name, var) in enumerate(variances.items(), start=1):
            sorted_indices = np.argsort(var)  # Get indices that would sort the variance
            sorted_variances = var[sorted_indices]  # Sort variances
            sorted_features = [self.idx2feature(idx) for idx in sorted_indices]  # Map indices to features
            fig.add_trace(
                go.Bar(
                    x=list(range(len(sorted_variances))),
                    y=sorted_variances,
                    text=sorted_features,  # Feature names
                    hovertemplate="Feature: %{text}<br>Variance: %{y}<extra></extra>",
                ),
                row=i,
                col=1,
            )

        # Update layout
        fig.update_layout(
            height=1500,  # Adjust height for better visualization
            title="Variance per Column for Each Matrix",
            showlegend=False,
        )

        # Show plot
        fig.show()

        return X_coverage, X_precision, X_pmi, X_geometric_mean, X_tfidf

class CorpusSyntacticFunctionsSinglet:
    def __init__(
        self,
        treebank_path: str,
        grew_pattern: str,
        patterns_text_file: str,
        use_sud: bool = False,
        min_occurrences: int = 10,
        matrix_type: str = "PMI"
    ):
        if use_sud:
            grewpy.set_config("sud")
        else:
            grewpy.set_config("ud")

        corpus = grewpy.Corpus(treebank_path)
        draft = grewpy.CorpusDraft(corpus)

        # all_matches is a dict with keys as matching nodes and values as a list of dictionaries of this kind of structure: 'chat': [{'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}]
        # for me this is because i clustered by lemma, so i get lemmas as keys
        all_matches = corpus.search(
            grewpy.Request(grew_pattern)
            .without("X[InIdiom=Yes]")
            .without("X[Idiom=Yes]")
            .without("X[InTitle=Yes]")
            .without("X[Title=Yes]")
            .without("X[Scrap=Yes]")
            .without("X[Foreign]")
            .without("X[Lang]")
            .without("X-[fixed]->Y")
            .without("Y-[flat:name]->X")
            .without("Y-[goeswith]->X"),
            clustering_parameter=["e.label", "X.upos", "Y.upos"],
        ) # getting rid of noisy matches

        sent_id_to_sentence = {
            draft[i].meta["sent_id"]: draft[i].features for i in range(len(draft)) # type: ignore
        }

        match_dep = {}
        for dep, x_poses in all_matches.items():
            for x_pos, y_poses in x_poses.items():
                for y_pos, matches in y_poses.items():
                    for m in matches:
                        if dep not in match_dep:
                            match_dep[dep] = []
                        if m not in match_dep[dep]:    # to avoid duplicates, because i count every relation twice, once from the perspective of X and once from the perspective of Y
                            match_dep[dep].append(m)

        copy_match_dep = match_dep.copy()
        for k, v in copy_match_dep.items():
            if len(v) < min_occurrences:
                match_dep.pop(k)
        with open(patterns_text_file) as in_stream:
            config = yaml.load(in_stream, Loader=yaml.Loader)

        templates = grex.utils.FeaturePredicate.from_config(config["templates"])
        feature_predicate = grex.utils.FeaturePredicate.from_config(
            config["features"], templates=templates
        )
        
        data = {k: list() for k in match_dep}
        for dep_unit, mts in match_dep.items():
            for match in mts:
                features = grex.data.extract_features(draft, match, feature_predicate)
                formatted_features = [
                            (
                                f"{':'.join(k)}={v}"
                                if not isinstance(v, set) # if the value is a set
                                else f"{':'.join(k)}={val}" # if the value is a set, we want to iterate through it and create a string for each value
                            )
                            for k, v in features.items()
                            for val in (v if isinstance(v, set) else [v])
                ]

                for feat in formatted_features:
                    if re.findall("Cxn", feat):
                        formatted_features.remove(feat)
                        
                data[dep_unit].append(formatted_features)

        unique_dep_units = sorted(set([k for k in data]))
        unique_features = sorted(
            set(
                [
                    feat
                    for _, match_dep in data.items() # iterating through (lemma, pos) : list of features for each match
                    for m in match_dep # iterating through each feature for the match
                    for feat in m
                ]
            )
        )

        self._idx2feature = {i: feat for i, feat in enumerate(unique_features)}
        self._feature2idx = {feat: i for i, feat in self._idx2feature.items()}
        self._idx2lexunit = {i: feat for i, feat in enumerate(unique_dep_units)}
        self._lexunit2idx = {feat: i for i, feat in self._idx2lexunit.items()}

        self.co_occurrence_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        for lexunit, feature_lists in data.items():
            for feature_list in feature_lists:
                for feature in feature_list:
                    self.co_occurrence_matrix[self._lexunit2idx[lexunit], self._feature2idx[feature]] += 1

        if matrix_type == "precision":
            self.make_precision_matrix(self.co_occurrence_matrix)
        elif matrix_type == "coverage":
            self.make_coverage_matrix(self.co_occurrence_matrix)
        elif matrix_type == "PMI":
            self.make_pmi_matrix(self.co_occurrence_matrix)
        elif matrix_type == "tf-idf":
            self.make_tfidf_matrix(self.co_occurrence_matrix)
        elif matrix_type == "geometric_mean":
            self.make_geometric_mean_matrix(self.co_occurrence_matrix)
        else:
            raise ValueError("Invalid matrix type. Choose from 'precision', 'coverage', 'PMI', or 'tf-idf'.")


    def idx2feature(self, idx: int) -> str:
        """Returns the feature name for a given index."""
        return self._idx2feature[idx]
    
    def feature2idx(self, feature: str) -> int:
        """Returns the index for a given feature name."""
        return self._feature2idx[feature]
    
    def idx2lexunit(self, idx: int) -> str:
        """Returns the lex unit name for a given index."""
        return self._idx2lexunit[idx]
    
    def lexunit2idx(self, lexunit: str) -> int:
        """Returns the index for a given lex unit name."""
        return self._lexunit2idx[lexunit]
    
    def make_precision_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the precision calculation: occurrences of word 1 with feature 1 / occurrences of all words with feature 1"""
        self.precision_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        column_sums = co_occurrence_matrix.sum(axis=0)
        self.precision_matrix = co_occurrence_matrix / column_sums
        # Avoid division by zero
        self.precision_matrix[np.isnan(self.precision_matrix)] = 0
        self.feature_matrix = self.precision_matrix

    def make_coverage_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the coverage calculation: occurrences of word 1 with feature 1 / occurrences of word 1"""
        self.coverage_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        self.coverage_matrix = co_occurrence_matrix / row_sums[:, np.newaxis]
        # Avoid division by zero
        self.coverage_matrix[np.isnan(self.coverage_matrix)] = 0
        self.feature_matrix = self.coverage_matrix

    def make_geometric_mean_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the geometric mean calculation: sqrt(precision * coverage)"""
        self.geometric_mean_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        self.precision_matrix = co_occurrence_matrix / co_occurrence_matrix.sum(axis=0)
        self.coverage_matrix = co_occurrence_matrix / co_occurrence_matrix.sum(axis=1)[:, np.newaxis]
        # Avoid division by zero
        self.precision_matrix[np.isnan(self.precision_matrix)] = 0
        self.coverage_matrix[np.isnan(self.coverage_matrix)] = 0
        self.geometric_mean_matrix = np.sqrt(
            self.precision_matrix * self.coverage_matrix
        )
        # Avoid division by zero
        self.feature_matrix = self.geometric_mean_matrix

    def make_pmi_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the PMI calculation: log( w1 with feature 1 * sum of co-occurrence matrix / sum of line * sum of column)"""
        self.pmi_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        column_sums = co_occurrence_matrix.sum(axis=0)
        total_sum = co_occurrence_matrix.sum()
        for i in range(len(self._idx2lexunit)):
            for j in range(len(self._idx2feature)):
                if co_occurrence_matrix[i, j] > 0:
                    self.pmi_matrix[i, j] = np.log(
                        (co_occurrence_matrix[i, j] * total_sum) /
                        (row_sums[i] * column_sums[j])
                    )
                else:
                    self.pmi_matrix[i, j] = 0  # Avoid log(0)

        self.feature_matrix = self.pmi_matrix
    
    def make_tfidf_matrix(self, co_occurrence_matrix):
        """Returns the matrix based on the tf-idf calculation: tf = occurrences of word 1 with feature 1 / occurrences of all other features for word 1 and idf = log(total number of lexical units / the number of lexical units with feature 1)"""
        self.tfidf_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        row_sums = co_occurrence_matrix.sum(axis=1)
        total_lexunits = co_occurrence_matrix.shape[0]
        column_sums = co_occurrence_matrix.sum(axis=0)
        for i in range(len(self._idx2lexunit)):
            for j in range(len(self._idx2feature)):
                if co_occurrence_matrix[i, j] > 0:
                    tf = co_occurrence_matrix[i, j] / row_sums[i]
                    idf = np.log(total_lexunits / column_sums[j])
                    self.tfidf_matrix[i, j] = tf * idf
                else:
                    self.tfidf_matrix[i, j] = 0
        self.feature_matrix = self.tfidf_matrix
        
    def show_variance_graphs(self):
        self.make_coverage_matrix(self.co_occurrence_matrix)
        X_coverage = self.feature_matrix

        self.make_precision_matrix(self.co_occurrence_matrix)
        X_precision = self.feature_matrix

        self.make_pmi_matrix(self.co_occurrence_matrix)
        X_pmi = self.feature_matrix

        self.make_geometric_mean_matrix(self.co_occurrence_matrix)
        X_geometric_mean = self.feature_matrix

        self.make_tfidf_matrix(self.co_occurrence_matrix)
        X_tfidf = self.feature_matrix

        # Calculate variance for each column in the matrices
        variances = {
            "Coverage": np.var(X_coverage, axis=0),
            "Precision": np.var(X_precision, axis=0),
            "PMI": np.var(X_pmi, axis=0),
            "TF-IDF": np.var(X_tfidf, axis=0),
            "Geometric Mean": np.var(X_geometric_mean, axis=0),
        }

        # Create subplots
        fig = make_subplots(rows=5, cols=1, subplot_titles=list(variances.keys()), shared_xaxes=True)

        # Add bar plots for each matrix
        for i, (name, var) in enumerate(variances.items(), start=1):
            sorted_indices = np.argsort(var)  # Get indices that would sort the variance
            sorted_variances = var[sorted_indices]  # Sort variances
            sorted_features = [self.idx2feature(idx) for idx in sorted_indices]  # Map indices to features
            fig.add_trace(
                go.Bar(
                    x=list(range(len(sorted_variances))),
                    y=sorted_variances,
                    text=sorted_features,  # Feature names
                    hovertemplate="Feature: %{text}<br>Variance: %{y}<extra></extra>",
                ),
                row=i,
                col=1,
            )

        # Update layout
        fig.update_layout(
            height=1500,  # Adjust height for better visualization
            title="Variance per Column for Each Matrix",
            showlegend=False,
        )

        # Show plot
        fig.show()

        return X_coverage, X_precision, X_pmi, X_geometric_mean, X_tfidf
    
class CorpusProsody:
    def __init__(
        self,
        treebank_path: str,
        grew_pattern: str,
        patterns_text_file: str,
        use_sud: bool = True,
        min_occurrences: int = 3,
        excluded_feature_patterns: list = None
    ):
        if use_sud:
            grewpy.set_config("sud")
        else:
            grewpy.set_config("ud")

        corpus = grewpy.Corpus(treebank_path)
        draft = grewpy.CorpusDraft(corpus)

        # all_matches is a dict with keys as matching nodes and values as a list of dictionaries of this kind of structure: 'chat': [{'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}]
        # for me this is because i clustered by lemma, so i get lemmas as keys
        all_matches = corpus.search(
            grewpy.Request(grew_pattern)
            .without("X[InIdiom=Yes]")
            .without("X[Idiom=Yes]")
            .without("X[InTitle=Yes]")
            .without("X[Title=Yes]")
            .without("X[Scrap=Yes]")
            .without("X[Foreign]")
            .without("X[Lang]")
            .without("X-[fixed]->Y")
            .without("Y-[flat:name]->X")
            .without("Y-[goeswith]->X"),
            clustering_parameter=["X.lemma"],
        ) # getting rid of noisy matches

        matches = {}
        for key, value in all_matches.items(): # type: ignore
            # remove those that have less than 10 occurrences
            if len(value) > min_occurrences:
                matches[key] = value

        # Create a dictionary to map sent_id to sentences for quick lookup
        # So for each sent_id I have each node and its features:  {'fr-ud-train_00001': {'0': {'form': '__0__'},'1': {'Definite': 'Def','Number': 'Plur', 'form': 'Les'}, '2': {'Gender': 'Fem','Number': 'Plur','form': 'commotions',}} etc
        sent_id_to_sentence = {
            draft[i].meta["sent_id"]: draft[i].features for i in range(len(draft)) # type: ignore
        }

        match_upos = {}
        # iterating through each lemma and its values like this: '€': [{'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}]
        for key, value in matches.items():
            # iterating through each match for the lemma so for example {'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}
            for m in value:
                match_sent_id = m["sent_id"]
                match_node_index = str(m["matching"]["nodes"]["X"])
                if match_sent_id in sent_id_to_sentence:
                    current_sentence_features = sent_id_to_sentence[match_sent_id]
                    if match_node_index in current_sentence_features.keys():
                        current_token_features = current_sentence_features[
                            match_node_index
                        ]
                        # if the token has a feature called ExtPos, we use that as the key, otherwise we use the upos feature
                        # ExtPos is used for example in 10% -> % has upos SYM but ExtPos Noun , so we want to use noun
                        # and what we're doing is creating a dictionary with keys (lemma, pos) and values as a list of matches
                        # so for example ('reason', 'NOUN') -> [match1, match2, match3] where a match looks like this: {'sent_id': 'fr-ud-train_11309', 'matching': {'nodes': {'X': '7'}, 'edges': {}}}
                       
                        if "ExtPos" in current_token_features:
                            match_upos.setdefault(
                                (key, current_token_features["ExtPos"]), []
                            ).append(m)
                        else:
                            try:
                                match_upos.setdefault(
                                    (key, current_token_features["upos"]), []
                                ).append(m)
                            except KeyError:
                                print(f"WARNING: Token with lemma '{key}' in sentence '{match_sent_id}' does not have a 'upos' feature.")
                            #     match_upos.setdefault(
                            #         (key, "NA"), []
                            #     ).append(m)

        # remove those that have less than 10 occurrences again because now we've split by pos so we might have some that are less than 10 
        # for example if we had reason with 11 occurrences before and now we have reason:VERB with 5 occurrences and reason:NOUN with 6 occurrences
        new_match_upos = {}
        for key, value in match_upos.items():
            if len(value) > min_occurrences:
                new_match_upos[key] = value
        match_upos = new_match_upos

        for key, value in match_upos.items():
            if len(value) < min_occurrences:
                print(f"WARNING: {key} has only {len(value)} occurrences but passed filtering!")
        
        # grex stuff
        with open(patterns_text_file) as in_stream:
            config = yaml.load(in_stream, Loader=yaml.Loader)

        templates = grex.utils.FeaturePredicate.from_config(config["templates"])
        feature_predicate = grex.utils.FeaturePredicate.from_config(
            config["features"], templates=templates
        )

        # data is a dict where each (lemma, pos) is a key and the value is an empty list for now
        data = {k: list() for k in match_upos}
        for lex_unit, mts in match_upos.items():
            for match in mts:
                features = grex.data.extract_features(draft, match, feature_predicate)
                # features looks like this: features={('node', 'X', 'own', 'Number'): 'Plur', ('node', 'X', 'own', 'rel_shallow'): 'comp:obj'
                formatted_features = [
                    (
                        f"{':'.join(k)}={v}"
                        if not isinstance(v, set) # if the value is a set
                        else f"{':'.join(k)}={val}" # if the value is a set, we want to iterate through it and create a string for each value
                    )
                    for k, v in features.items()
                    for val in (v if isinstance(v, set) else [v])
                ]
                # so what we did is go from features = {('key1', 'key2'): 'value1', ('key3',): {'value2', 'value3'}} to ['key1:key2=value1', 'key3=value2', 'key3=value3']
                # and now we append the features to the data dict for the corresponding (lemma, pos) key
                if excluded_feature_patterns:
                    formatted_features = self._filter_features(formatted_features, excluded_feature_patterns)
                data[lex_unit].append(formatted_features)

        unique_lemma = sorted(set([k for k in data]))
        unique_features = sorted(
            set(
                [
                    feat
                    for _, match_upos in data.items() # iterating through (lemma, pos) : list of features for each match
                    for m in match_upos # iterating through each feature for the match
                    for feat in m
                ]
            )
        )

        new_unique_features = []
        for feat in unique_features:
            if re.search("phoneticform", feat):
                continue
            elif re.search("Gloss=", feat):
                continue
            elif re.search("Nucleus", feat):
                continue
            elif re.search(":Onset", feat):
                continue
            elif re.search("SylForm", feat):
                continue

            if re.search(r"\d.?\d$", feat):
                numberless_feat = feat.split("=")[0]
                new_unique_features.append(numberless_feat)
            else:
                new_unique_features.append(feat)
            
        new_unique_features = list(set(new_unique_features))

        self._idx2feature = {i: feat for i, feat in enumerate(new_unique_features)}
        self._feature2idx = {feat: i for i, feat in self._idx2feature.items()}
        self._idx2lexunit = {i: feat for i, feat in enumerate(unique_lemma)}
        self._lexunit2idx = {feat: i for i, feat in self._idx2lexunit.items()}

        self.feature_matrix = np.zeros((len(self._idx2lexunit), len(self._idx2feature)))
        n_samples_total = len(match_upos)
        for lexunit, feature_lists in data.items():
            row_idx = self._lexunit2idx[lexunit]
            numerical_feats = defaultdict(float)
            for feature_list in feature_lists:
                for feature in feature_list:
                    try:
                        feature_name, value = feature.split("=")
                    except ValueError:
                        print(f"Skipping malformed feature: {feature}")
                        continue
                    if feature_name in new_unique_features:
                        numerical_feats[feature_name] += float(value)
                    elif feature in new_unique_features:
                        self.feature_matrix[row_idx, self._feature2idx[feature]] += 1
            self.feature_matrix[row_idx] = self.feature_matrix[row_idx] / n_samples_total
            current_n_samples = len(feature_lists)
            for feat_name, total_value in numerical_feats.items():
                if feat_name in self._feature2idx:
                    col_idx = self._feature2idx[feat_name]
                    if self.feature_matrix[row_idx, col_idx] == 0:
                        self.feature_matrix[row_idx, col_idx] = total_value / current_n_samples

    def idx2feature(self, idx: int) -> str:
        """Returns the feature name for a given index."""
        return self._idx2feature[idx]
    
    def feature2idx(self, feature: str) -> int:
        """Returns the index for a given feature name."""
        return self._feature2idx[feature]
    
    def idx2lexunit(self, idx: int) -> str:
        """Returns the lex unit name for a given index."""
        return self._idx2lexunit[idx]
    
    def lexunit2idx(self, lexunit: str) -> int:
        """Returns the index for a given lex unit name."""
        return self._lexunit2idx[lexunit]
    
    def _filter_features(self, features, excluded_patterns):
        """Filter out features matching any of the excluded patterns."""
        import re
        filtered_features = []
        for feature in features:
            should_exclude = False
            for pattern in excluded_patterns:
                if re.search(pattern, feature):
                    should_exclude = True
                    break
            if not should_exclude:
                filtered_features.append(feature)
        return filtered_features