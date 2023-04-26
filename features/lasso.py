import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso


class LassoSelection:
    def __init__(self, dataset, target_col, alpha=0.1, fit_intercept=True, normalize=False, max_iter=1000, tol=0.0001):
        self.dataset = dataset
        self.target_col = target_col
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
        self.pipeline = None
        self.num_cols = None
        self.cat_cols = None
        self.feature_cols = None
        self.X = None
        self.y = None
        self._determine_col_type()

    def fit(self):
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder()

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.num_cols),
                ('cat', categorical_transformer, self.cat_cols)
            ])

        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', Lasso(alpha=self.alpha, fit_intercept=self.fit_intercept,
                                                            max_iter=self.max_iter,
                                                            tol=self.tol))])
        self.pipeline.fit(self.X, self.y)

    def transform(self):
        return self.pipeline.transform(self.X)

    def fit_transform(self):
        self.fit()
        return self.transform()

    def get_selected_features(self):
        mask = self.pipeline.named_steps['regressor'].coef_ != 0
        selected_features = [f for f, m in zip(self.X.columns, mask) if m]
        return selected_features

    def _determine_col_type(self):
        # Identify numerical columns
        num_cols = self.dataset.select_dtypes(include=["int", "float"]).columns
        num_cols = num_cols.drop(self.target_col, errors='ignore')
        self.num_cols = num_cols.to_list()

        # Identify categorical columns
        cat_cols = self.dataset.select_dtypes(include=["object", "bool"]).columns
        cat_cols = cat_cols.drop(self.target_col, errors='ignore')
        self.cat_cols = cat_cols.to_list()

        self.feature_cols = self.num_cols + self.cat_cols
        self.X = self.dataset[self.feature_cols]
        self.y = self.dataset[self.target_col]
