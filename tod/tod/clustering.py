from abc import ABC
from .corpus import Corpus
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import numpy as np


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
        )  # TODO: check these parameters
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
