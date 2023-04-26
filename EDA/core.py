import numpy as np
import csv

from . import statistics
from libs.utils import files
from .visualisation import DataVisualisation

import matplotlib.pyplot as plt


class EDA:
    def __init__(self, issue_data, target_col):
        print(f'\n________________START of EDA________________\n')
        self.issue_data = issue_data
        self.target_col = target_col
        self.outliers = None
        self.days_iqr = None
        self.num_cols = None
        self.cat_cols = None
        self._clean_data()

    def detect_del_outliers(self, method='none'):
        outlier_cols = self.target_col + self.num_cols

        # Number of observations in the data (used later to calculated outlier ratio)
        obersvations = len(self.issue_data.index)
        outliers = {}

        for col in outlier_cols:
            # Calculate IQR, lower quantile and upper quantile for column
            iqr, q1, q3 = statistics.calc_iqr_score(self.issue_data[col])

            # Assign values (outliers) to mask
            mask = (self.issue_data[col] < (q1 - 1.5 * iqr)) | (self.issue_data[col] > (q3 + 1.5 * iqr))

            # store the outlier index and value in a tuple
            outlier = [(index, value) for index, value in self.issue_data[mask][col].items()]

            # add the column name as key and the outlier list as value
            outliers[col] = outlier

            if method == 'cap':
                self.issue_data.loc[mask, col] = np.max(self.issue_data[col][~mask])
            elif method == 'drop':
                outlier_index = self.issue_data[mask].index
                self.issue_data = self.issue_data.drop(outlier_index)

        if method == 'cap':
            print('Outliers are capped to the maximum value that is not considered as outlier by IQR')
        elif method == 'none':
            print('Outliers are still in data')
        elif method == 'drop':
            ratio = (len(self.issue_data.index) / obersvations) * 100
            formatted_ratio = "{:.2f}%".format(ratio)
            print(f"Outliers removed from data, {formatted_ratio}% of observations removed")

        self.outliers = outliers
        return self.issue_data

    def plot_outliers(self, outlier_method):
        outlier_cols = self.target_col + self.num_cols

        high_values = ['days_till_fix', 'description_length']
        low_values = [item for item in outlier_cols if item not in high_values]

        # Create a list of sets of columns to be plotted together
        column_sets = [high_values, low_values]

        obersvations = len(self.issue_data.index)

        for i, column_set in enumerate(column_sets):
            # Create a new figure for each set of columns
            fig, axes = plt.subplots(nrows=len(column_set), ncols=1, figsize=(15, 9))

            for j, col in enumerate(column_set):
                # Calculate the quartile values
                iqr, q1, q3 = statistics.calc_iqr_score(self.issue_data[col])

                # Identify outliers using the IQR method
                mask = (self.issue_data[col] < (q1 - 1.5 * iqr)) | (self.issue_data[col] > (q3 + 1.5 * iqr))
                outliers = self.issue_data[mask]

                # Create the boxplot
                axes[j].boxplot(self.issue_data[col], showfliers=True, vert=False)
                # Plot the outliers as individual points
                axes[j].scatter(outliers[col], [1 for _ in range(len(outliers))], c='red', s=50)
                # Add a label for the y-axis
                axes[j].set_yticks([1], [col])
                # Rotate y labels by 90 degrees
                axes[j].set_yticklabels([col], rotation=90)

                # Add outliers as % of observations to plot
                outlier_percentage = (len(outliers) / obersvations) * 100
                axes[j].text(0.05, 0.95, f'Mean: {self.issue_data[col].mean():.2f}',
                             transform=axes[j].transAxes,
                             fontsize=12, verticalalignment='top')
                axes[j].text(0.20, 0.95, f'Std: {self.issue_data[col].std():.2f}'
                             , transform=axes[j].transAxes,
                             fontsize=12, verticalalignment='top')
                axes[j].text(0.35, 0.95,
                             f'Outliers: {outlier_percentage:.2f}%', transform=axes[j].transAxes,
                             fontsize=12, verticalalignment='top')

            plt.savefig(f'{files.project_folder()}/eda/figures/boxplots/boxplots_set{i + 1}.png',
                        dpi=600,
                        transparent=False)

        return self.issue_data

    def remove_target_outliers(self):
        iqr, q1, q3 = statistics.calc_iqr_score(self.issue_data['days_till_fix'])
        self.days_iqr = [iqr, q1, q3]

        # Remove all data where the outlier for days till fix is greater than 150
        self.issue_data = self.issue_data.loc[
            (self.issue_data['days_till_fix'] <= q3) & (self.issue_data['days_till_fix'] > q1)]
        print(f'Outliers have been removed for days till fix at 75th percentile with max day {q3}')

        return self.issue_data

    def cat_value_counts(self, replace, percentage):
        value_counts = self._value_counts()
        with open(f"{files.project_folder()}/cat_value_counts.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(value_counts)
        print(value_counts)

        if replace == 'yes':
            observations = len(self.issue_data)
            threshold = int(len(self.issue_data) * percentage / 100)
            self.issue_data = self._replace_small_categories(threshold=threshold)
            print(f'Category values of categorical Features with categories smaller than {percentage}% '
                  f'of data have been replaced by "MISSING"')
        else:
            print('Categorical Features with small categories have not been replaced')
        return self.issue_data

    def upset_plot(self):
        #TODO:
        # Not in use ATM, boolean data does not apply for dataset
        adj_data = self.cat_value_counts(replace='true', percentage=5)
        cat_data = adj_data[self.cat_cols + self.target_col]

        # Based on interpretability drop a few columns
        cat_data = cat_data.drop(['transition', 'status', 'author_active', 'reporter', 'assignee'], axis=1)

        # Remove rows where at least 3 categories are missing (less interpretable):
        cat_data = cat_data[(cat_data == 'MISSING').sum(axis=1) < 3]

        upset_data = cat_data.set_index(list(cat_data.columns[:-1]))

        # UpSetPlot Not Finished
        pass

    def visualise_data(self):
        print(f'\n________________START of Data Visualisation________________\n')
        outliers = self.detect_del_outliers()
        visualise = DataVisualisation(dataset=self.issue_data, target=self.target_col)
        # visualise.plot_all_distribution(num_cols=self.num_cols, cat_cols=self.cat_cols)
        # visualise.correlation_heatmap(num_cols=self.num_cols, cat_cols=self.cat_cols)
        pass

    def _value_counts(self):
        return [self.issue_data[col].value_counts() for col in self.issue_data[self.cat_cols]]

    def _replace_small_categories(self, threshold=50):
        # Set threshold to that at least 10% of observations should have this category
        for col in self.issue_data[self.cat_cols]:
            value_counts = self.issue_data[col].value_counts()
            small_categories = value_counts[value_counts < threshold].index
            self.issue_data[col].replace(small_categories, 'MISSING', inplace=True)
        return self.issue_data

    def _clean_data(self):
        # Drop Missing Data
        self._drop_missing()

        # Replace empty strings
        self.issue_data.replace('', np.nan, inplace=True)

        # Mutate Null values
        self._mutate_missing()

        # Drop data for testing
        self._drop_test_data()

    def _drop_missing(self):
        # Check for missing data
        missing_data = self.issue_data.isnull().sum()
        print(f"Missing Data per Feature: \n{missing_data}\n")

        # Select data with more than 25% of total values missing
        missing_25 = missing_data[missing_data > (0.25 * len(self.issue_data))]
        self.issue_data.drop(missing_25.index, axis=1, inplace=True)
        print(f"Dropped Features due to more than 25% Missing Data: \n{missing_25.index.to_list()}\n")

    def _drop_test_data(self):
        self.issue_data = self.issue_data.dropna(subset=['resolution_date'])

    def _mutate_missing(self):
        self.issue_data = self.issue_data.set_index(['key', 'id'])

        # Identify numerical columns with missing values
        num_cols = self.issue_data.select_dtypes(include=["int", "float"]).columns
        num_cols = num_cols.drop(self.target_col, errors='ignore')
        self.num_cols = num_cols.to_list()

        cat_cols = self.issue_data.select_dtypes(include=["object", "bool"]).columns
        cat_cols = cat_cols.drop(self.target_col, errors='ignore')
        self.cat_cols = cat_cols.to_list()

        missing_num_cols = self.issue_data[num_cols].columns[self.issue_data[num_cols].isna().any()].tolist()
        missing_cat_cols = self.issue_data[cat_cols].columns[self.issue_data[cat_cols].isnull().any()].tolist()

        # Create a dictionary of means for each key value
        key_means = self.issue_data.groupby('key')[missing_num_cols].mean()

        # Fill keys that have no value at all for columns with NaN with mean of all keys for that column
        key_means.fillna(key_means.mean(), inplace=True)

        # Update missing numeric_cols with mean values for the respective keys
        self.issue_data[missing_num_cols] = self.issue_data[missing_num_cols].combine_first(key_means)
        print(f"Mutated missing numerical variables with mean of corresponding KEY")

        # Create a new category 'MISSING' for missing categorical values
        self.issue_data[missing_cat_cols] = self.issue_data[missing_cat_cols].fillna('MISSING')
        print(f"Mutated missing categorical variables with new category 'MISSING'\n")
