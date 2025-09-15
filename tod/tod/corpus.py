import grewpy
import yaml
import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(1, "/Users/madalina/Documents/M2TAL/stage/grex/grex2")
import pyximport

pyximport.install()
import grex.data
import grex.utils
import grex.features


class Corpus:
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
                for f, feature_value in features.items():
                    if f[3] == "lemma":
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
