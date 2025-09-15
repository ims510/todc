from abc import ABC
from sklearn.neighbors import LocalOutlierFactor
from .corpus import Corpus
import numpy as np
from typing import Any
from .pyod.sod import SOD2


class OutlierDetector(ABC):
    y_pred: np.ndarray
    scores: np.ndarray
    outliers: list
    inliers: list


class LOF(OutlierDetector):
    """
    Local Outlier Factor (LOF) algorithm for outlier detection.

    This class implements the LOF algorithm, which identifies outliers based on the local density of data points.
    It computes the local reachability density and compares it to the local reachability density of its neighbors.

    Attributes:
        n_neighbors (int): Number of neighbors to use for computing the local reachability density.
        contamination (float): Proportion of outliers in the data set.
    """

    def __init__(
        self,
        corpus: Corpus,
        n_neighbors: int = 5,
        contamination: float = 0.1,
    ):

        clf = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
        )
        self.y_pred = clf.fit_predict(corpus.feature_matrix)
        self.scores = clf.negative_outlier_factor_

        self.outliers = []
        self.inliers = []
        for i in range(len(self.y_pred)):
            if self.y_pred[i] == -1:
                self.outliers.append(corpus.idx2lexunit(i))
            else:
                self.inliers.append(corpus.idx2lexunit(i))


class SOD(OutlierDetector):
    def __init__(
        self,
        corpus: Corpus,
        n_neighbours: int = 5,
        contamination: float = 0.1,
        ref_set: int = 2,
    ):
        model = SOD2(
            n_neighbors=n_neighbours,
            contamination=contamination,
            ref_set=ref_set,
        )
        model.fit(corpus.feature_matrix)
        self.y_pred = np.array(model.predict(corpus.feature_matrix))
        self.scores = model.decision_scores_

        self.relevant_features = model.relevant_features_
        self.weights = model.relevant_feature_weights_

        self.outliers = []
        self.inliers = []
        for i in range(len(self.y_pred)):
            if self.y_pred[i] == 1:
                self.outliers.append(corpus.idx2lexunit(i))
            else:
                self.inliers.append(corpus.idx2lexunit(i))
