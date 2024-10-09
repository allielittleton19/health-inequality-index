import statistics
import numpy as np
import math
from numpy import array, exp
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

import matplotlib.pyplot as mp
import pandas as pd
import seaborn as sb
import numpy as np
import statsmodels.api as sm
import matplotlib.cm as cm


def run_simple_regression(data, curr_var, X, y):

    # regressor = LinearRegression()
    # regressor.fit(X, y)

    # predictions = regressor.predict(X)
    X = data[curr_var]
    y = data['Standard Deviation Index Value']
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2, missing='drop')
    est2 = est.fit()
    return est2.pvalues[1], est2.rsquared


def regression(data):
    vars = data.columns
    n_cols = len(vars)

    y = data[vars[1]].values.reshape(-1, 1)

    p_vals = []
    rsquared_vals = []
    for i in range(1, n_cols):
        curr_var = vars[i]
        X = data[curr_var].values.reshape(-1, 1)
        p, r_sq = run_simple_regression(data, curr_var, X, y)
        p_vals.append(p)
        rsquared_vals.append(r_sq)

    d = {'Variable': vars[1:], 'p': p_vals, 'R-squared': rsquared_vals}

    df = pd.DataFrame(d)
    df.to_excel("Univariate Regression Table.xlsx")

def correlogram(data):
    # creating mask
    mask = np.triu(np.ones_like(data.corr()))

    # plotting a triangle correlation heatmap
    dataplot = sb.heatmap(data.corr(), annot=True, mask=mask)

    # displaying heatmap
    mp.show()

def impute_vals(df):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputed_data = imp.fit_transform(df)
    df = pd.DataFrame(data=imputed_data, columns=df.columns)
    return df


def random_forest(df, name, y_str, vars_to_drop):
    data = impute_vals(df)

    vars_to_drop.append(y_str)

    print(vars_to_drop)

    y = data[y_str]
    X = data.drop(vars_to_drop, axis=1)

   # X = data.drop([y_str, 'Life Expectancy', 'Under-5 Mortality'], axis=1)
    # X = data.drop(['Standard Deviation Index Value'], axis=1)

    # Train a random forest regressor on data
    rf = RandomForestRegressor(n_estimators=10000, random_state=42)
    rf.fit(X, y)

    # Get feature importances
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    feature_importances_df = pd.DataFrame({
        'Feature': X.columns[indices],
        'Importance': importances[indices],
        'Std': std[indices]
    })
    feature_importances_df.to_excel(f"{name}_feature_importances.xlsx", index=False)

    # drop most important feature (comment out if want all features plotted)
    # indices = indices[1:]

    # only plot top 15 most important features (comment out if want all features plotted)
    indices = indices[:15]

    # Plot the feature importances
    plt.figure(figsize=(12, 10))
    plt.title("Feature importances")
    plt.bar(range(len(indices)), importances[indices], yerr=std[indices], align="center")
    # Rotate the x-axis labels to 45 degrees and optionally adjust the font size if needed.
    plt.xticks(range(len(indices)), X.columns[indices], rotation=45, ha='right', fontsize='small')
    plt.xlim([-1, len(indices)])
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.savefig(name)  # Save the figure to a file

    feature_ranking = {X.columns[i]: rank + 1 for rank, i in enumerate(indices)}
    return feature_ranking


def lasso(df):
    data = impute_vals(df)
    y = data['Standard Deviation Index Value']
    X_unscaled = data.drop(['Standard Deviation Index Value', 'Under-5 Mortality', 'Life Expectancy'], axis=1)
    variable_names = X_unscaled.columns.tolist()

    # Standardize the variables
    scaler = StandardScaler()
    X = scaler.fit_transform(X_unscaled)

    # Create a LASSO model
    lasso = Lasso(alpha=0.1)

    # Fit the model to your data
    lasso.fit(X, y)

    # Access the coefficient magnitudes
    coefficient_magnitudes = np.abs(lasso.coef_)

    print(coefficient_magnitudes)

    # Rank variables by importance
    variable_importance = sorted(zip(variable_names, coefficient_magnitudes), key=lambda x: x[1], reverse=True)
    sorted_variable_names, sorted_variable_importance = zip(*variable_importance)

    # Create a bar graph of variable importance
    plt.figure(figsize=(8, 6))
    plt.bar(sorted_variable_names, sorted_variable_importance)
    plt.xticks(rotation='vertical')
    plt.xlabel('Variable')
    plt.ylabel('Importance')
    plt.title('Variable Importance - LASSO')
    plt.tight_layout()
    plt.show()


def compare_ranks(combined_df):

    combined_df['gini_rank'] = combined_df['gini'].rank(ascending=False)
    combined_df['blfmm_rank'] = combined_df['Standard Deviation Index Value'].rank(ascending=False)
    combined_df['jamison_rank'] = combined_df['Jamison Index'].rank(ascending=False)


    combined_df[['Country', 'gini_rank', 'blfmm_rank', 'jamison_rank']].to_excel("rank_comparison.xlsx")

    # plot index comparisons of blfmm and gini
    plt.figure(figsize=(10, 6))
    plt.scatter(combined_df['Standard Deviation Index Value'], combined_df['gini'], color='blue', s=100)
    plt.title('Gini vs BLFMM Value')
    plt.xlabel('BLFMM Value')
    plt.ylabel('Gini Index Value')
    plt.grid(True)
    plt.savefig("Value_Comparison.png")

    # plot index comparisons of blfmm and jamison
    plt.figure(figsize=(10, 6))
    plt.scatter(combined_df['Standard Deviation Index Value'], combined_df['Jamison Index'], color='blue', s=100)
    plt.title('Jamison vs BLFMM Value')
    plt.xlabel('BLFMM Value')
    plt.ylabel('Jamison Index Value')
    plt.grid(True)
    plt.savefig("BLFMM_Jamison_Value_Comparison.png")

    # plot index comparisons of gini and jamison
    plt.figure(figsize=(10, 6))
    plt.scatter(combined_df['gini'], combined_df['Jamison Index'], color='blue', s=100)
    plt.title('Jamison vs Gini Value')
    plt.xlabel('Gini Value')
    plt.ylabel('Jamison Index Value')
    plt.grid(True)
    plt.savefig("Gini_Jamison_Value_Comparison.png")


    # plot rank comparisons of blfmm and gini
    plt.figure(figsize=(10, 6))
    plt.scatter(combined_df['blfmm_rank'], combined_df['gini_rank'], color='blue', s=100)
    plt.title('Gini vs BLFMM Ranks')
    plt.xlabel('BLFMM Rank')
    plt.ylabel('Gini Rank')
    plt.grid(True)
    plt.savefig("BLFMM_Gini_Rank_Comparison.png")

    # plot rank comparisons of blfmm and jamison
    plt.figure(figsize=(10, 6))
    plt.scatter(combined_df['blfmm_rank'], combined_df['jamison_rank'], color='blue', s=100)
    plt.title('Jamison vs BLFMM Ranks')
    plt.xlabel('BLFMM Rank')
    plt.ylabel('Jamison Rank')
    plt.grid(True)
    plt.savefig("BLFMM_Jamison_Rank_Comparison.png")

    # plot rank comparisons of gini and jamison
    plt.figure(figsize=(10, 6))
    plt.scatter(combined_df['gini_rank'], combined_df['jamison_rank'], color='blue', s=100)
    plt.title('Gini vs Jamison Ranks')
    plt.xlabel('Gini Rank')
    plt.ylabel('Jamison Rank')
    plt.grid(True)
    plt.savefig("Gini_Jamison_Rank_Comparison.png")


def reformat_dataframe(df, year, index):
    # Pivot the DataFrame
    pivoted_df = df.pivot_table(
        index=['Country', 'Code'],  # Set both 'Country Name' and 'Code' as the index
        columns='Series Name',  # Spreads each unique 'Series Name' across the columns
        values=f'{year} [YR{year}]'  # Fills the table with values from '2019 [YR2019]'
    )

    # The DataFrame is now indexed by both Country Name and Code, each 'Series Name' is a column
    # Display the first few rows to confirm the structure
    print(pivoted_df.head())

    # add index in there
    df = pd.read_excel("jamison_blfmm_gini.xlsx")
    df_2019 = df[df['Year'] == year]
    index_2019 = df_2019[['Code', f'{index}', 'Infant Mortality']]

    pivoted_df.reset_index(inplace=True)

    pivoted_df = pd.merge(pivoted_df, index_2019, on='Code', how='left')

    extra_covariates_df = pd.read_excel(f"Input_DFs/Inequality Index Values_{year}.xlsx")

    additional_covariates = [
        'Country',  # Keep 'Country' for merging
        'Modal Age',
        'World Bank Income Classification',
        'Under-5 Mortality',
        'Gini Index',
        'Democracy Index'
    ]

    extra_covariates_df = extra_covariates_df[additional_covariates]

    final_updated_df = pd.merge(pivoted_df, extra_covariates_df, on='Country', how='left')

    final_updated_df.to_excel(f'updated_pivoted_data_with_{index}_85%.xlsx', index=True)
    print(final_updated_df.head())

    return final_updated_df


def run_rf_index(index, vars_to_drop, dir):
    df = pd.read_excel(f"updated_pivoted_data_with_{index}_85%.xlsx")
    df.drop('Country', axis=1, inplace=True)
    df.drop('Code', axis=1, inplace=True)

    himr_data = df[df["Modal Age"] == 0]
    limr_data = df[df["Modal Age"] != 0]

    himr_data.drop('Modal Age', axis=1, inplace=True)
    limr_data.drop('Modal Age', axis=1, inplace=True)
    df.drop('Modal Age', axis=1, inplace=True)

    loc = f'RFs/RFs 191 Covariates/{dir}/{index}'

    print(vars_to_drop)

    feature_ranking_all = random_forest(df, f'{loc}/RF_{index}_no_IM_all.png', index, vars_to_drop)
    feature_ranking_himr = random_forest(himr_data, f'{loc}/RF_{index}_no_IM_HIMR.png', index, vars_to_drop)
    feature_ranking_limr = random_forest(limr_data, f'{loc}/RF_{index}_no_IM_LIMR.png', index, vars_to_drop)


def main():
    years = range(2010, 2020)

    himr_rankings_by_year = {}
    limr_rankings_by_year = {}

    '''
    df = pd.read_excel("WDI/covariates_list_85%_collinearity.xlsx")
    df = reformat_dataframe(df, 2019, 'Jamison Index')
    '''


    vars_to_drop_1 = ["Life expectancy at birth, total (years)", "Infant Mortality"]
    vars_to_drop_2 = ["Infant Mortality"]
    vars_to_drop_3 = []

    run_rf_index("Jamison Index", vars_to_drop_1, "No Infant Mortality or LE")
    run_rf_index("Jamison Index", vars_to_drop_2, "No Infant Mortality")
    run_rf_index("Jamison Index", vars_to_drop_3, "All vars")




    '''
    for year in years:
        df = pd.read_excel("Input_DFs/Inequality Index Values_" + str(year) + ".xlsx")
        df.drop(df.columns[0], axis=1, inplace=True)

        df.drop('Country', axis=1, inplace=True)
        df.drop('WHO Region', axis=1, inplace=True)
        df.drop('Absolute Deviation Index Value', axis=1, inplace=True)

        himr_data = df[df["Modal Age"] == 0]
        limr_data = df[df["Modal Age"] != 0]
        limr_rankings_by_year[year] = random_forest(limr_data, 'RFs/RF_LIMR_' + str(year) + '.png', "Standard Deviation Index Value")
        himr_rankings_by_year[year] = random_forest(himr_data, 'RFs/RF_HIMR_' + str(year) + '.png', "Standard Deviation Index Value")

    df_himr = pd.DataFrame(himr_rankings_by_year)
    df_limr = pd.DataFrame(limr_rankings_by_year)

    df_himr.to_excel("HIMR_rankings_by_year.xlsx")
    df_limr.to_excel("LIMR_rankings_by_year.xlsx")
    '''

    '''
    # comparing Ivan's values for 2019
    gini_df = pd.read_excel("Ivan_values_2019.xlsx")
    blfmm_df = pd.read_excel("Inequality Index Values_2019+code.xlsx")
    jamison_df = pd.read_excel("Jamison_Indices_2019.xlsx")

    temp_df = pd.merge(blfmm_df, gini_df, on='Code')
    combined_df = pd.merge(temp_df, jamison_df, on='Code')

    combined_df.to_excel("combined_df2.xlsx")

    compare_ranks(combined_df)
    '''

    '''
    # run RF on Ivan vals

    combined_df.drop('Country', axis=1, inplace=True)
    combined_df.drop('country', axis=1, inplace=True)
    combined_df.drop('WHO Region', axis=1, inplace=True)
    combined_df.drop('Absolute Deviation Index Value', axis=1, inplace=True)
    combined_df.drop('Code', axis=1, inplace=True)
    combined_df.drop('Standard Deviation Index Value', axis=1, inplace=True)
    combined_df.to_excel('final_combined_df.xlsx')


    gini_df = combined_df.copy()
    gini_df.drop('sd', axis=1, inplace=True)
    himr_data = gini_df[gini_df["Modal Age"] == 0]
    limr_data = gini_df[gini_df["Modal Age"] != 0]

    sd_df = combined_df.copy()
    sd_df.drop('gini', axis=1, inplace=True)

    gini_rankings_by_year = {}
    sd_rankings_by_year = {}

    gini_rankings_by_year[2019] = random_forest(limr_data, 'RF_gini_LIMR.png', "gini")

    # sd_rankings_by_year[2019] = random_forest(sd_df, 'RF_sd_no_LE.png', "sd")

    df_gini = pd.DataFrame(gini_rankings_by_year)
    # df_sd = pd.DataFrame(sd_rankings_by_year)

    df_gini.to_excel("Ivan_gini_rankings_2019.xlsx")
    # df_sd.to_excel("Ivan_sd_rankings_2019.xlsx")

#plot_spaghetti(himr_rankings_by_year, 'HIMR Feature Rankings Over Time', 'HIMR_Sphaghetti.png')
   # plot_spaghetti(limr_rankings_by_year, 'LIMR Feature Rankings Over Time', 'LIMR_Sphaghetti.png')

    #correlogram(himr_data)
    #correlogram(limr_data)
    #regression(himr_data)
    #regression(limr_data)

    #lasso(himr_data)
'''

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
