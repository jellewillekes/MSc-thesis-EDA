import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plots

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.kernel_approximation import Nystroem

from libs.utils import files


class PCA_Analysis:
    def __init__(self, n_components=6, pca_type='rbf'):
        self.n_components = n_components
        self.pca_type = pca_type
        self.model = None
        self.transformed_X = None
        print(f"Perform PCA Analysis on numerical columns using {pca_type} PCA")

    def find_best_n_components(self, X):
        # First use Z-score standardization to standardize all numerical input variables:
        X = self._z_score_standardise(X)

        explained_variance = None
        cum_explained_variance = None
        if self.pca_type == 'standard':
            self._fit_standard_pca(X)
            explained_variance = self.model.explained_variance_ratio_
            cum_explained_variance = np.cumsum(explained_variance)
        elif self.pca_type == 'rbf':
            eigen_vals, _ = np.linalg.eigh(rbf_kernel(X, X))
            eigen_vals = eigen_vals[::-1]
            explained_variance = eigen_vals / np.sum(eigen_vals)
            cum_explained_variance = np.cumsum(explained_variance)

        plots.scree_plot(cum_explained_variance, self.pca_type)

    def feature_components_corr(self, X):
        self.fit_transform(X)

        # Get the correlation matrix between the original features and the transformed components
        corr_matrix = np.corrcoef(X.T, self.transformed_X.T)
        # Get the correlation between the features and the components
        corr_with_components = corr_matrix[:X.shape[1], X.shape[1]:]
        # Convert the correlation array to a dataframe
        corr_df = pd.DataFrame(np.abs(corr_with_components),
                               index=X.columns,
                               columns=[f"PC{i}" for i in range(1, self.n_components + 1)])
        plots.plot_heatmap(corr_df, title=f'Heatmap features and components of {self.pca_type}')
        print(f"Correlation matrix between features and components of {self.pca_type}")
        print(corr_df)

    def fit_transform(self, X):
        if self.pca_type == 'standard':
            self._fit_standard_pca(X)
        elif self.pca_type == 'rbf':
            self._fit_rbf_pca(X)

    def _fit_standard_pca(self, X):
        self.model = PCA(n_components=self.n_components)
        self.transformed_X = self.model.fit_transform()
        return self.model.fit(X)

    def _fit_rbf_pca(self, X, gamma=None):
        K = rbf_kernel(X, X, gamma)
        eigen_vals, eigen_vecs = np.linalg.eigh(K)
        eigen_vals = eigen_vals[::-1]
        eigen_vecs = eigen_vecs[:, ::-1]
        if self.n_components is not None:
            eigen_vals = eigen_vals[:self.n_components]
            eigen_vecs = eigen_vecs[:, :self.n_components]
        self.transformed_X = np.dot(eigen_vecs, np.diag(np.sqrt(eigen_vals)))
        return self.transformed_X

    def _z_score_standardise(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean)/std

