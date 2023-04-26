from data.processing import DataProcessor
from eda.core import EDA

from features.core import FeatureEngineering
from models.core import ModelSelector
from models.evaluate import ModelEvaluator

from sklearn.model_selection import train_test_split

from libs.utils import files
import json


if __name__ == '__main__':
    DataProcessor = DataProcessor()
    data = DataProcessor.process_data()

    target_col = ['days_till_fix']

    EDA = EDA(data, target_col)
    # dataset = EDA.remove_target_outliers()
    dataset = EDA.detect_del_outliers(method='drop')

    # If replacing uncommon categories with 'MISSING', uncomment next line:
    # data_adjusted = EDA.cat_value_counts(replace='yes', percentage=5)

    features = FeatureEngineering(dataset, target_col)

    my_cats = ['assignee', 'reporter', 'priority', 'issue_type', 'timezone_reporter', 'author']
    my_nums = ['watch_count', 'comment_count', 'description_length', 'log_size']
    my_features = my_cats + my_nums

    lasso_features = features.perform_lasso()

    X = dataset[my_features]
    y = dataset['days_till_fix']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Specify the numerical and categorical features
    num_features = my_nums  # List of numerical feature names
    cat_features = my_cats  # List of categorical feature names

    # Define the grid search parameters
    rt_params = {
        'n_estimators': [10],
        'max_depth': [5, 10],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }

    rf_params = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    # Initialize the ModelSelector
    model = ModelSelector(num_features, cat_features, model_name='RandomForest', grid_params=rt_params)

    # Fit the model on the training data
    model.fit(X, y)

    with open(f'{files.project_folder()}/models/settings/best_params_RandomForest.json', 'r') as f:
        best_params = json.load(f)

    print(best_params)

    # Make predictions on the test data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    evaluate = ModelEvaluator(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred)

    evaluate.evaluate_regression()

   #TODO write code that used best params for fitting and predictiong (do not predict every run):
"""    if model_name == 'GradientBoosting':
        model = GradientBoostingRegressor(**best_params)
    elif model_name == 'RandomForest':
        model = RandomForestRegressor(**best_params)
    else:
        print('ERROR: The model_name is not part of the set of models to choose from. Please choose from the '
              'following models: GradientBoosting, RandomForest, LSTM')"""

    # Compute the bias and variance of the model
    # model.bias_variance(X_test, y_test)




