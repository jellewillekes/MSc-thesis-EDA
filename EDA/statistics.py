import numpy as np
import pandas as pd
import scipy.stats as stats


def calc_z_score(data: pd.Series):
    return (data - np.mean(data)) / np.std(data)


def calc_iqr_score(data: pd.Series):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    return iqr, q1, q3


def binwidth_scott(data: pd.Series):
    return 3.5 * data.std() / (len(data) ** (1 / 3))


def binwidth_freedman(data: pd.Series):
    iqr, _, _ = calc_iqr_score(data)
    return 2 * iqr ** (-1 / 3)


def summary_stats(data: pd.Series):
    # calculate summary stats for data
    mean = data.mean()[0]
    median = data.median()[0]
    mode = data.mode().loc[0][0]
    standard_deviation = data.std()[0]
    skewness = data.skew()[0]
    kurtosis = data.kurt()[0]

    # print the results
    print("Summary Statistics Target Variable:\n")
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Mode: {mode:.2f}")
    print(f"Standard Deviation: {standard_deviation}")
    print(f"Skewness: {skewness:.2f}\tNormal: 0.00\tPoisson: 0.00")
    print(f"Kurtosis: {kurtosis:.2f}\tNormal: 3.00\tPoisson: 3.00\n")
    pass


def test_normal_distribution(data: pd.Series):
    print('Test if target variable follows a NORMAL distribution:')

    def Shapiro(data):
        # perform the Shapiro-Wilk test
        W, p_value = stats.shapiro(data)
        # check if the p-value is greater than 0.05
        if p_value < 0.05:
            print("\tShapiro-Wilk: Reject the null hypothesis, NOT NORMAL distribution.")
        else:
            print("\tShapiro-Wilk: Fail to reject the null hypothesis, NORMAL distribution.")
        pass

    def Anderson(data):
        # perform the Anderson-Darling test
        A2, critical_values, sig_level = stats.anderson(data)
        # check if the test statistic is less than the critical value at the desired significance level
        if A2 < critical_values[-1]:
            print("\tAnderson-Darling: Fail to reject the null hypothesis, NORMAL distribution.")
        else:
            print("\tAnderson-Darling: Reject the null hypothesis, NOT NORMAL distribution.")

    def JarqueBera(data):
        t_stat, p_value = stats.jarque_bera(data)

        if p_value < 0.05:
            print("\tJarque-Bera: Reject the null hypothesis, NOT NORMAL distribution.")
        else:
            print("\tJarque-Bera: Fail to reject the null hypothesis, NORMAL distribution.\n")

    Shapiro(data['days_till_fix'])
    JarqueBera(data['days_till_fix'])
    Anderson(data['days_till_fix'])
    pass


def test_poisson_distribution(data: pd.Series):
    # estimate lambda using the sample mean
    lamb = np.mean(data['days_till_fix'])

    # perform the Kolmogorov-Smirnov test
    D, p_value = stats.kstest(data['days_till_fix'], "poisson", args=(lamb,))

    # check if the p-value is less than 0.05
    if p_value < 0.05:
        print("\tKolmogorov-Smirnov G.O.F.: Reject the null hypothesis, NOT POISSON distribution.\n")
    else:
        print("\tKolmogorov-Smirnov G.O.F.: Fail to reject the null hypothesis, Poisson distribution.\n")


def calc_cramers_v(cat_data: pd.DataFrame):
    cat_count = cat_data.nunique()
    N = len(cat_data)
    V = np.zeros((cat_count.size, cat_count.size))
    for i in range(cat_count.size):
        for j in range(cat_count.size):
            contingency = pd.crosstab(cat_data.iloc[:, i], cat_data.iloc[:, j])
            chi2 = stats.chi2_contingency(contingency)[0]
            phi2 = chi2 / N
            V[i, j] = np.sqrt(phi2 / min(cat_count[i], cat_count[j]))
    V = pd.DataFrame(V)
    V.index = cat_count.index
    V.columns = cat_count.index
    return V


def ANOVA_categories_test(dataset, target_variable, cat_variable):
    categories = dataset[cat_variable].unique()
    N = len(categories)

    # Create a list of target variable values for each category
    groups = [dataset[dataset[cat_variable] == category][target_variable] for category in categories]

    # results = pd.DataFrame(index=categories, columns=categories)

    G = 0
    S = 0
    C = 0
    N = 0

    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            # Perform the ANOVA test between two categories
            f_statistic, p_value = stats.f_oneway(groups[i], groups[j])
            if p_value < 0.05:
                # REJECT NULL of no difference: The means of the target variable across categories is significantly
                # different.
                t_statistic, t_p_value = stats.ttest_ind(groups[i], groups[j])
                if t_p_value < 0.05:
                    # REJECT NULL of no difference: Difference between means is considered significant.
                    if t_statistic > 0:
                        # Direction of difference, here 'G' indicated that category i is Greater than category j
                        G += 1
                        # results.loc[categories[i], categories[j]] = 'G'

                    else:
                        # Direction of difference, here 'S' indicated that category i is Smaller than category j
                        S += 1
                        # results.loc[categories[i], categories[j]] = 'S'

                else:
                    # ACCEPT NULL of no difference: Difference between means is considered insignificant.
                    C += 1
                    # results.loc[categories[i], categories[j]] = 'C'

            else:
                # ACCEPT NULL of no difference: The means of the target variable across categories are not
                # significantly different.
                N += 1
                # results.loc[categories[i], categories[j]] = 'N'

    diff_mean = G + S
    total = G + S + C + N
    perc_diff = diff_mean / total
    return perc_diff * 100

    # results.to_csv(f"{files.project_folder()}/eda/tables/heatmap_categories_{cat_variable}.csv")

    # TODO:
    # Did not manage to convert string to rgb colors that can be plotted with either SNS or IMSHOW.
    # Dataframe calculation works, but results in warnings for calculating statistics.

    # if N < 1:
        #color_palette = {'G': 0.65, 'S': 0.67, 'C': 0.75, 'N': 0.82, np.nan: 0.99}
        #numeric_results = results.applymap(lambda x: color_palette[x])
        #sns.heatmap(numeric_results, cmap='gist_ncar', cbar=True,
        #            cbar_kws={'ticks': [0, 0.25, 0.40, 0.5, 0.65, 0.7, 0.8, 1]})
        #plt.savefig(f'{files.project_folder()}/eda/figures/features/heatmap_categories_{cat_variable}.png', dpi=600,
        #            transparent=False)
        #plt.show()
        #plt.close()


