from abc import ABC
import numpy as np
from .corpus import Corpus
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class DimensionReduction(ABC):
    def __init__(self):
        self.reduced_matrix: np.ndarray

class Pca_corpus(DimensionReduction):
    def __init__(self, corpus: Corpus, n_components: int = 2):
        pca = PCA(n_components=n_components)
        self.reduced_matrix = pca.fit_transform(corpus.feature_matrix)
        self.explained_variance = pca.explained_variance_ratio_

class Tsne_corpus(DimensionReduction):
    def __init__(self, corpus: Corpus, n_components: int = 2, random_state: int = 42):
        tsne = TSNE(n_components=n_components, random_state=random_state)
        self.reduced_matrix = tsne.fit_transform(corpus.feature_matrix)

class Pca_matrix(DimensionReduction):
    def __init__(self, x, n_components: int = 2):
        pca = PCA(n_components=n_components)
        self.reduced_matrix = pca.fit_transform(x)
        self.explained_variance = pca.explained_variance_ratio_

class Tsne_matrix(DimensionReduction):
    def __init__(self, x, n_components: int = 2, random_state: int = 42):
        tsne = TSNE(n_components=n_components, random_state=random_state)
        self.reduced_matrix = tsne.fit_transform(x)