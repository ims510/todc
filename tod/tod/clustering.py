from abc import ABC
from .corpus import Corpus
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans as SKLearnKMeans
import numpy as np
import pysparcl

class Clustering(ABC):
    def __init__(self):
        self._cluster2lexunit: dict = {}
        self._lexunit2cluster: dict = {}
        self.clusters: dict = {}

    def _generate_cluster2lexunit(self, clusters: dict, corpus: Corpus):
        """Generates the lex units for each cluster."""
        self._cluster2lexunit = {}
        for cluster, lexunits in clusters.items():
            self._cluster2lexunit[cluster] = [corpus.idx2lexunit(i) for i in lexunits]

    def _generate_lexunit2cluster(self, clusters: dict, corpus: Corpus):
        """Generates the cluster for each lex unit."""
        self._lexunit2cluster = {}
        for cluster, lexunits in clusters.items():
            for lexunit_idx in lexunits:
                self._lexunit2cluster[corpus.idx2lexunit(lexunit_idx)] = cluster

    def cluster2lexunit(self, cluster: int) -> list:
        """Returns the lex units for a given cluster."""
        return self._cluster2lexunit.get(cluster, [])
    
    def lexunit2cluster(self, lexunit: str) -> int:
        """Returns the cluster for a given lex unit."""
        return self._lexunit2cluster.get(lexunit, -1)


class HierarchicalClustering(Clustering):
    def __init__(self, corpus: Corpus, max_clusters: int = 50):
        super().__init__()
        optimal_clusters, silhouette_scores = self.find_optimal_clusters(
            corpus, max_clusters
        )
        distance_matrix = pdist(corpus.feature_matrix, metric="cosine")
        self.linked = linkage(
            distance_matrix, method="complete", optimal_ordering=True
        )  
        labels = fcluster(self.linked, optimal_clusters, criterion="maxclust")
        self.clusters = {i: [] for i in range(1, optimal_clusters + 1)}
        for i, label in enumerate(labels):
            self.clusters[label].append(i)

        self._generate_cluster2lexunit(self.clusters, corpus)
        self._generate_lexunit2cluster(self.clusters, corpus)

    def find_optimal_clusters(
        self, corpus: Corpus, max_clusters: int = 50, metric="cosine", method="complete"
    ):
        distance_matrix = pdist(corpus.feature_matrix, metric=metric)  # type: ignore
        linked = linkage(distance_matrix, method=method, optimal_ordering=True)

        silhouette_scores = []
        for num_clusters in range(2, max_clusters + 1):
            labels = fcluster(linked, num_clusters, criterion="maxclust")
            if len(np.unique(labels)) > 1:  # Ensure there is more than one cluster
                score = silhouette_score(corpus.feature_matrix, labels, metric=metric)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(-1)  # Append a low score if only one cluster

        optimal_clusters = (
            np.argmax(silhouette_scores) + 2
        )  # +2 because range starts from 2
        return optimal_clusters, silhouette_scores

class KMeans(Clustering):
    def __init__(self, corpus: Corpus, k: int = 10, n_init: int = 20):
        super().__init__()
 

        kmeans = SKLearnKMeans(n_clusters=k, init='random', n_init=n_init).fit(corpus.feature_matrix)
        labels = kmeans.labels_
        self.labels = labels
        self.clusters = {i: [] for i in range(k)}
        for i, label in enumerate(labels):
            self.clusters[label].append(i)

        self._generate_cluster2lexunit(self.clusters, corpus)
        self._generate_lexunit2cluster(self.clusters, corpus)

class DBScan(Clustering):
    """
    Minimum samples (“MinPts”): the fewest number of points required to form a cluster
    ε (epsilon or “eps”): the maximum distance two points can be from one another while still belonging to the same cluster
    """

    def __init__(self, corpus: Corpus, eps: float = 0.1, min_samples: int = 2):
        super().__init__()
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        labels = db.fit_predict(corpus.feature_matrix)
        self.clusters = {}
        for i, label in enumerate(labels):
            if label not in self.clusters:
                self.clusters[label] = []
            self.clusters[label].append(i)

        self._generate_cluster2lexunit(self.clusters, corpus)
        self._generate_lexunit2cluster(self.clusters, corpus)

class SparseHierarchical(Clustering):
    def __init__(self, corpus: Corpus, max_clusters: int = 50, method="average", metric="cosine"):
        super().__init__()

        optimal_clusters, silhouette_scores, perm, result, weights = self.find_optimal_clusters(
            corpus, max_clusters, method=method, metric=metric
        )
 
        self.linked = linkage(
            result["u"], method=method, optimal_ordering=True
        )  
        labels = fcluster(self.linked, optimal_clusters, criterion="maxclust")
        self.clusters = {i: [] for i in range(1, optimal_clusters + 1)}
        for i, label in enumerate(labels):
            self.clusters[label].append(i)

        self._generate_cluster2lexunit(self.clusters, corpus)
        self._generate_lexunit2cluster(self.clusters, corpus)

        important_features = np.argsort(-weights)
        for i in important_features[:5]:
            print(f"Feature {corpus._idx2feature[i]}: weight {weights[i]:.3f}")
    def find_optimal_clusters(
        self, corpus: Corpus, max_clusters: int = 50, metric="cosine", method="average"
    ):
        
        perm = pysparcl.hierarchy.permute(corpus.feature_matrix)
        best_weight_bound = perm['bestw']
        result = pysparcl.hierarchy.pdist(
            corpus.feature_matrix, wbound=best_weight_bound
        )
        distance_matrix = result["u"]
        weights = result["w"]
        linked = linkage(distance_matrix, method=method, optimal_ordering=True)

        silhouette_scores = []
        for num_clusters in range(2, max_clusters + 1):
            labels = fcluster(linked, num_clusters, criterion="maxclust")
            if len(np.unique(labels)) > 1:  # Ensure there is more than one cluster
                score = silhouette_score(corpus.feature_matrix, labels, metric=metric)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(-1)  # Append a low score if only one cluster
        
        optimal_clusters = (
            np.argmax(silhouette_scores) + 2
        )  # +2 because range starts from 2
        return optimal_clusters, silhouette_scores, perm, result, weights
    
class SparseKMeans(Clustering):
    def __init__(self, corpus: Corpus, k: int = 10, nperms: int = 25, nvals: int = 10, top_n_features = None, cluster_defining_features: bool = True):
        super().__init__()
 
        perm = pysparcl.cluster.permute(corpus.feature_matrix, k=k, nperms=nperms, nvals=nvals)
        best_weight_bound = perm['bestw']
        result = pysparcl.cluster.kmeans(corpus.feature_matrix, k=k, wbounds=best_weight_bound)[0]
        weights = result['ws']

        if cluster_defining_features:
            self.get_cluster_defining_features(corpus, result, top_n_features)
        else:
            feature_importance = []
            for feat_idx in range(len(weights)):
                feature_name = corpus.idx2feature(feat_idx)
                feature_weight = weights[feat_idx]
                feature_importance.append((feat_idx, feature_name, feature_weight))

            feature_importance_sorted = sorted(feature_importance, key=lambda x: x[2], reverse=True)

            all_features_printing = False
            if top_n_features == None or top_n_features > len(feature_importance_sorted):
                top_n_features = len(feature_importance_sorted)
                all_features_printing = True
            if all_features_printing:
                print("All features ranked by importance:")
            else:
                print(f"Top {top_n_features} features ranked by importance:")
            print("=" * 80)
            for rank, (feat_idx, feature_name, feature_weight) in enumerate(feature_importance_sorted[:top_n_features], start=1):
                print(f"{rank:3d}. {feature_name:<50} weight: {feature_weight:.6f}")

        cluster_assignments = result['cs']
        self.clusters = {i: [] for i in range(k)}
        for i, cluster_idx in enumerate(cluster_assignments):
            self.clusters[cluster_idx].append(i) 

        self._generate_cluster2lexunit(self.clusters, corpus)
        self._generate_lexunit2cluster(self.clusters, corpus)

    def get_cluster_defining_features(self, corpus, result, top_n_features, weight_threshold=0.0):
        """
        Extract the defining features for each cluster from a sparse k-means result.
        
        Parameters
        ----------
        x : np.ndarray
            Data matrix of shape (n_samples, n_features).
        results : list of dict
            Output from the `kmeans` function. Each dict must contain:
            - 'ws': feature weights (array of length n_features)
            - 'cs': cluster assignments (array of length n_samples)
        top_n : int, optional
            Number of top features to return per cluster (default=10).
        result_index : int, optional
            Index of which result in the list to use (default=-1, i.e. the last one).
        weight_threshold : float, optional
            Minimum absolute weight for a feature to be considered (default=0.0).

        Returns
        -------
        cluster_features : dict
            Dictionary mapping cluster index → pandas DataFrame with columns:
            ['feature_index', 'weight', 'centroid_value', 'global_mean', 'importance'].
            Sorted by importance (descending).
        """
        # Select the desired result
        ws = np.array(result['ws'])
        cs = np.array(result['cs'])
        x = corpus.feature_matrix
        cooc = corpus.co_occurrence_matrix

        n_clusters = len(np.unique(cs))
        n_features = x.shape[1]
        
        # Optional normalization of weights
        if np.sum(np.abs(ws)) > 0:
            ws = ws / np.sum(np.abs(ws))
        
        # Filter by threshold
        if weight_threshold > 0:
            ws_mask = ws > weight_threshold
        else:
            ws_mask = np.ones_like(ws, dtype=bool)
        
        C_f = cooc.sum(axis=0) # global counts per feature
        N = C_f.sum()   # total co-ocurrences globally
        
        # # Compute centroids and global mean
        # centroids = np.zeros((k, p))
        # for j in range(k):
        #     centroids[j, :] = x[cs == j].mean(axis=0)
        # global_mean = x.mean(axis=0)
        
        # Build output
        cluster_features = {}
        for j in range(n_clusters):
            in_cluster = cs == j # boolean mask for lex units in cluster j
            n_j = cooc[in_cluster, :].sum()  # total co-ocurrences in cluster j

            importance = np.zeros(n_features) # making the importance array that we will fill in

            for f in range(n_features):
                if not ws_mask[f]:
                    continue
                
                # Frequency = lex units in cluster j that have feature f / total lex units in cluster j
                f_values_in_cluster = x[in_cluster, f] # a 1d array of the values of feature f for lex units in cluster j (e.g. [0.2, 0.6, 0, 0.1 ...])
                # if f_values_in_cluster > 0 then the lex unit has the feature so i'd have [1, 1, 0, 1 ...] for the example above
                freq_fj = (f_values_in_cluster > 0).sum() / in_cluster.sum() 

                # Stability = inverse of 1 + variance within the cluster
                stab_fj = 1 / (1 + np.var(f_values_in_cluster))

                # Contrast = p(f|j) / p(f)
                c_fj = cooc[in_cluster, f].sum() 
                contrast_fj = 0
                if C_f[f] > 0 and n_j > 0:
                    p_f_given_j = c_fj / n_j
                    p_f_global = C_f[f] / N
                    contrast_fj = p_f_given_j / p_f_global
                
                # Importance
                importance[f] = ws[f] * freq_fj * stab_fj * contrast_fj

            top_idx = np.argsort(importance)[::-1][:top_n_features]
            feature_names = [corpus.idx2feature(idx) for idx in top_idx]
            top_importance = importance[top_idx]

            
            print("=" * 80)
            print(f"Cluster {j} - Top {len(top_idx)} defining features:")
            print("=" * 80)
            for rank, (name, imp) in enumerate(zip(feature_names, top_importance), 1):
                print(f"{rank:3d}. {name:<50} importance: {imp:.6f}")
            print()

            cluster_features[j] = list(zip(top_idx, feature_names, top_importance))

        return cluster_features
