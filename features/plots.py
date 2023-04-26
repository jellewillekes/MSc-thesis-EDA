import matplotlib.pyplot as plt
import scipy.stats as stats

import numpy as np

from libs.utils import files


def qq_plot(data, col):
    plt.figure()
    stats.probplot(data[col], dist='norm', plot=plt)
    plt.title(f"QQ-Plot for Target and Normal")
    plt.savefig(f'{files.project_folder()}/eda/figures/qqplot/qqplot_{col}.png', dpi=600,
                transparent=False)
    plt.close()


def scree_plot(cum_explained_variance, pca_type):
    plt.plot(range(1, len(cum_explained_variance) + 1), cum_explained_variance)
    plt.title(f"Scree plot for {pca_type} PCA")
    plt.xlim(0, 15)
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.savefig(f"{files.project_folder()}/features/figures/pca_elbow_{pca_type}.png")
    plt.close()


def plot_heatmap(X, title):
    plt.imshow(X.values, cmap='coolwarm')
    plt.xticks(np.arange(len(X.columns)), X.columns)
    plt.yticks(np.arange(len(X.index)), X.index, rotation=30)
    plt.title(title)
    plt.colorbar()
    plt.subplots_adjust(bottom=0.2, left=0.2)
    plt.savefig(f'{files.project_folder()}/features/figures/{title}.png', dpi=600,
                transparent=False)
    plt.close()
