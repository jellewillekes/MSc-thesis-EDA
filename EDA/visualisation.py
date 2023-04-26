import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

from libs.utils import files
from . import statistics


class DataVisualisation:
    def __init__(self, dataset, target, max_categories=10):
        self.dataset = dataset
        self.max_categories = max_categories
        self.target = target

    def plot_all_distribution(self, num_cols, cat_cols):
        print('Plot distributions for features and target variable:')
        print('Histograms for Numerical Variables plotted with Binwitdh according to Scott')
        for col in self.dataset[num_cols + self.target]:
            self.histogram_plot(col)
        for col in self.dataset[cat_cols]:
            self.barplot_plot(col)

    def histogram_plot(self, col):
        # Calculate the bin width by using Scott's rule
        data = self.dataset[col]
        bin_width = statistics.binwidth_scott(data)
        # Create the bins via an arrange function
        bins = np.arange(data.min(), data.max() + bin_width, bin_width)
        plt.figure()
        plt.hist(self.dataset[col], bins=bins, edgecolor='black', alpha=0.7)
        plt.xlabel(col, fontsize=8)
        plt.xticks(rotation=30)
        plt.ylabel('Count')
        plt.title(f'Histogram of {col}')
        plt.savefig(f'{files.project_folder()}/eda/figures/distributions/histogram_{col}.png', dpi=600,
                    transparent=False)
        plt.close()

    def barplot_plot(self, col):
        mean_days = self.dataset.groupby(col)[self.target].mean()
        cat_counts = self.dataset[col].value_counts()

        # Only plot categories with highest counts, else plots are not feasible
        if len(cat_counts) > self.max_categories:
            cat_counts = pd.concat([cat_counts[:self.max_categories],
                                    pd.Series(cat_counts[self.max_categories:].sum(),
                                              index=['Other'])])
            count_plot = cat_counts.plot(kind='bar', edgecolor='black')
            self._add_text_on_barplot(count_plot, col, cat_counts, mean_days)
        # For features with less than max categories, plot all categories
        else:
            count_plot = cat_counts.plot(kind='bar', edgecolor='black')
            self._add_text_on_barplot(count_plot, col, cat_counts, mean_days)

        count_plot.set_xlabel(col, fontsize=8)
        count_plot.set_ylabel('Count')
        count_plot.set_title(f'Bar Plot of {col}')
        plt.subplots_adjust(bottom=0.2, top=0.8)
        plt.xticks(rotation=30)
        plt.savefig(f'{files.project_folder()}/eda/figures/distributions/barplot_{col}.png', dpi=600,
                    transparent=False)
        plt.close()

    def correlation_heatmap(self, num_cols, cat_cols):
        corr_num = self.dataset[num_cols + self.target].corr()
        corr_cat = statistics.calc_cramers_v(self.dataset[cat_cols])

        corr_num = self.dataset[num_cols + self.target].corr()
        corr_cat = statistics.calc_cramers_v(self.dataset[cat_cols])

        fig, (ax_num, ax_cat) = plt.subplots(1, 2, figsize=(10, 5))
        plt.suptitle(f"Heatmap for Features", fontsize=16)
        # Plot numerical correlation matrix
        im1 = ax_num.imshow(corr_num, cmap='coolwarm')
        ax_num.set_title('Numerical Features', fontsize=12)
        ax_num.set_xticks(np.arange(corr_num.shape[0]))
        ax_num.set_yticks(np.arange(corr_num.shape[0]))
        ax_num.set_xticklabels(num_cols + self.target, fontsize=8, rotation=45)
        ax_num.set_yticklabels(num_cols + self.target, fontsize=8)
        for i in range(len(num_cols) + 1):
            for j in range(len(num_cols) + 1):
                ax_num.annotate("{:.2f}".format(corr_num.iloc[i, j]), (j, i), ha="center", va="center", color="white")
        plt.colorbar(im1, ax=ax_num)

        # Plot categorical correlation matrix
        im2 = ax_cat.imshow(corr_cat, cmap='coolwarm')
        ax_cat.set_title('Categorical Features', fontsize=12)
        ax_cat.set_xticks(np.arange(corr_cat.shape[0]))
        ax_cat.set_yticks(np.arange(corr_cat.shape[0]))
        ax_cat.set_xticklabels(cat_cols, fontsize=8, rotation=45)
        ax_cat.set_yticklabels(cat_cols, fontsize=8)
        plt.colorbar(im2, ax=ax_cat)

        fig.subplots_adjust(wspace=0.5, left=0.15)
        plt.savefig(f'{files.project_folder()}/eda/figures/features/heatmap.png', dpi=600,
                    transparent=False)
        plt.close()

    def upset_plot(self, cat_cols):
        cat_data = self.dataset[cat_cols]

        # Create a sample dataframe
        df = pd.DataFrame({'A': ['A1', 'A2', 'A3'], 'B': ['B2', 'B3', 'B4'], 'C': ['C3', 'C4', 'C5']})

        # Convert the columns to categorical data
        df = df.astype('category')

        # Convert the dataframe to an features format
        # features = UpSet(df, intersection_plot_elements=True)

        # Show the features plot
        # features.show()

        pass

    def _add_text_on_barplot(self, plot, col, cat_counts, mean_days):
        for i, value in enumerate(cat_counts.index):
            if value == 'Other':
                highest_cats = cat_counts[:self.max_categories].index
                mean_cat = self.dataset[~self.dataset[col].isin(highest_cats)][self.target].mean()[0]
                plot.text(i, 1.01 * cat_counts.loc[value], f'{mean_cat:.2f}',
                          ha='center',
                          va='bottom',
                          fontsize=8,
                          color='black',
                          label='Mean Days')
            else:
                mean_cat = mean_days.loc[value][0]
                plot.text(i, 1.01 * cat_counts.loc[value], f'{mean_cat:.2f}',
                          ha='center',
                          va='bottom',
                          fontsize=8,
                          color='black',
                          label='Mean Days')