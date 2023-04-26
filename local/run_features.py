from eda.core import EDA
from features.core import FeatureEngineering
from data.processing import DataProcessor


if __name__ == '__main__':
    DataProcessor = DataProcessor()
    data = DataProcessor.process_data()

    target_col = ['days_till_fix']

    EDA = EDA(data, target_col)
    # dataset = EDA.remove_target_outliers()
    dataset = EDA.detect_del_outliers(method='cap')

    # If replacing uncommon categories with 'MISSING', uncomment next line:
    # data_adjusted = EDA.cat_value_counts(replace='yes', percentage=5)

    features = FeatureEngineering(dataset, target_col)
    features.target_distribution()

    # WARNING: Running Category Importance calculations by ANOVA takes some time. Many Calculations
    # features.category_importance_ANOVA()
    my_cats = ['assignee', 'reporter', 'priority', 'issue_type', 'timezone_reporter', 'author']

    features.numerical_importance_PCA(pca_type='rbf')
    my_nums = ['watch_count', 'comment_count', 'description_length', 'log_size']
    my_features = [my_cats + my_nums]

    lasso_features = features.perform_lasso()

    print('test')











