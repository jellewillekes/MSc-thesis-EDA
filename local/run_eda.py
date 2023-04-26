from eda.core import EDA
from data.processing import DataProcessor

import data

if __name__ == '__main__':
    DataProcessor = DataProcessor()
    data = DataProcessor.process_data()

    target_col = ['days_till_fix']

    EDA = EDA(data, target_col)
    # EDA.plot_outliers(outlier_method='none')
    # EDA.detect_outliers(outlier_method='none')
    EDA.remove_target_outliers()
    # data_adjusted = EDA.cat_value_counts(replace='yes', percentage=5)

    # EDA.upset_plot()

    # EDA.visualise_data()





