import pandas as pd

from . import plots
from .lasso import LassoSelection
from .pca import PCA_Analysis

from eda import statistics
from libs.utils import files


class FeatureEngineering:
    def __init__(self, issue_data, target_col):
        print(f'\n________________START of Feature Engineering________________\n')
        self.issue_data = issue_data
        self.target_col = target_col
        self.num_cols = None
        self.cat_cols = None
        self._determine_col_type()

    def target_distribution(self):
        print(f'\n________________Target Distribution________________\n')
        target = self.issue_data[self.target_col]

        # Print some summary stats for target variable
        statistics.summary_stats(target)
        statistics.test_normal_distribution(target)
        statistics.test_poisson_distribution(target)
        plots.qq_plot(data=self.issue_data, col='days_till_fix')
        print("Conclusion: QQ plot and Anderson-Darling suggest that target variable follows a Normal distribution")
        print("On the other hand, Shapiro and Jarque-Bera Reject the Null and sugges does NOT follow a Normal "
              "distribution")
        pass

    def category_importance_ANOVA(self):
        # Warning: RUNNING takes a while, many calculations.
        print(f"ANOVA category importance in Categorical Variables")
        print(f"Test if the means of the target variable across categories are significantly different.")
        print(f"Below the % of categories with significantly different means is printed per Categorical Variable")
        # Get number of unique categories per categorical variable:
        unique_cats = self.issue_data[self.cat_cols].nunique().sort_values(ascending=True)
        # Only use categories with at least 2 categories (trivial)
        sorted_cols = unique_cats[unique_cats > 2].index.to_list()

        results = {}
        for col in self.issue_data[sorted_cols]:
            print(f"TEST: {col}")
            perc_diff_mean = statistics.ANOVA_categories_test(dataset=self.issue_data, target_variable=self.target_col,
                                                              cat_variable=col)
            results[col] = perc_diff_mean
            print(f"Category {col}: {perc_diff_mean:.2f}%")

        results = pd.DataFrame(list(results.items()), columns=['Feature', 'Perc. Diff. Mean Target across Categories'])
        results.iloc[:, [1]] = round(results.iloc[:, [1]], 2)
        results = results.sort_values(by=results.columns[1], ascending=False)
        results.to_csv(f"{files.project_folder()}/eda/tables/categorical_feature_importance.csv")
        print(results)

    def numerical_importance_PCA(self, pca_type):
        num_data = self.issue_data[self.num_cols]
        pca = PCA_Analysis(pca_type=pca_type)
        # pca.find_best_n_components(num_data)
        pca.feature_components_corr(num_data)

        print('From the diagonal line in the Scree plot, 6 principal components best explain the variance in the data')

    def perform_lasso(self):
        lasso = LassoSelection(dataset=self.issue_data, target_col=self.target_col)
        lasso.fit()
        selected_features = lasso.get_selected_features()

        print(f'Selected features by Lasso: {selected_features}')
        return selected_features

    def _determine_col_type(self):
        # Identify numerical columns with missing values
        num_cols = self.issue_data.select_dtypes(include=["int", "float"]).columns
        num_cols = num_cols.drop(self.target_col, errors='ignore')
        self.num_cols = num_cols.to_list()

        cat_cols = self.issue_data.select_dtypes(include=["object", "bool"]).columns
        cat_cols = cat_cols.drop(self.target_col, errors='ignore')
        self.cat_cols = cat_cols.to_list()


