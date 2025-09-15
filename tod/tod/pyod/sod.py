# -*- coding: utf-8 -*-
"""Subspace Outlier Detection (SOD)
"""
# Author: Yahya Almardeny <almardeny@gmail.com>
# License: BSD 2 clause

import numpy as np

from pyod.models.sod import SOD

class SOD2(SOD):

    def _sod(self, X):
        ref_inds = self._snn(X)
        anomaly_scores = np.zeros(shape=(X.shape[0],))
        self.relevant_features_ = []  # Store relevant features for each sample
        self.relevant_feature_weights_ = []  # Store relevant feature weights for each sample
        self.reference_points_ = []  # Store reference points for each point
        for i in range(X.shape[0]):
            obs = X[i]
            ref = X[ref_inds[i,],]
            means = np.mean(ref, axis=0)
            var_total = np.sum(np.sum(np.square(ref - means))) / self.ref_set
            var_expect = self.alpha * var_total / X.shape[1]
            var_actual = np.var(ref, axis=0)
            var_inds = [1 if (j < var_expect) else 0 for j in var_actual]
            rel_dim = np.sum(var_inds)
            weights = np.square(obs - means)
            if rel_dim != 0:
                anomaly_scores[i] = np.sqrt(
                    np.dot(var_inds, weights) / rel_dim)
            self.relevant_features_.append(var_inds)  # Save relevant features
            self.relevant_feature_weights_.append(weights)
            self.reference_points_.append(ref) # Save reference points

        return anomaly_scores
