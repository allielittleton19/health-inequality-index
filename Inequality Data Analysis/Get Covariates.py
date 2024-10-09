import pandas as pd
import numpy as np


def filter_years(df, pct):
    df = df.drop(columns=['Code'])

    # Calculate the percentage of missing data for each year
    missing_percentage = df.isnull().mean() * 100

    # Filter years with ≤10% missing data
    valid_years = missing_percentage[missing_percentage <= pct].index.tolist()



def pivot_dataframe(df):
    df = df.groupby(['Code', 'Year'], as_index=False).mean()

    pivoted_df = df.pivot(index='Code', columns='Year', values='Electoral democracy index')

    pivoted_df.reset_index(inplace=True)

    return pivoted_df


def test_missingness_individual_covariate():

    # read in individual covariate dataframe
    # (data should be in same format as world bank dataframe)
    df = pd.read_csv("WDI/electoral-democracy-index.csv")

    df = pivot_dataframe(df)

    large_countries = pd.read_excel('Populations.xlsx')
    large_country_codes = list(large_countries["Code"])

    filtered_df = df[df["Code"].isin(large_country_codes)]

    # pass in percent missingness
    filter_years(filtered_df, 10)
    filter_years(filtered_df, 20)
    filter_years(filtered_df, 30)
    filter_years(filtered_df, 40)


def replace_missing_values(df):
    # Replace ".." with NaN
    df.replace("..", np.nan, inplace=True)
    return df


def test_collinearity(df, missingness, collinearity_param):

    correlation_matrix = df.corr()

    # Use np.triu to mask the lower triangle and diagonal of the correlation matrix
    mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    filtered_corr = correlation_matrix.where(mask)

    # Stack the matrix to collapse it into a series with a MultiIndex
    stacked_corr = filtered_corr.stack()

    stacked_corr.index.rename(['Series Name 1', 'Series Name 2'], inplace=True)

    stacked_corr_df = stacked_corr.reset_index(name='Correlation')


    significant_pairs = stacked_corr_df[
        (stacked_corr_df['Correlation'].abs() >= collinearity_param) &
        (stacked_corr_df['Series Name 1'] != stacked_corr_df['Series Name 2'])  # Avoid self-comparison
        ]


    # Create a set to keep track of variables to drop
    vars_to_drop = set()

    col_1 = []
    col_2 = []

    for _, row in significant_pairs.iterrows():
        var1, var2 = row['Series Name 1'], row['Series Name 2']
        if var1 not in vars_to_drop and var2 not in vars_to_drop:
            # Compare missingness for the two variables
            if missingness[var1] > missingness[var2]:
                vars_to_drop.add(var1)
            elif missingness[var2] > missingness[var1]:
                vars_to_drop.add(var2)
            # if same missingness, add them to spreadsheet to manually compare
            else:
                col_1.append(var1)
                col_2.append(var2)

    # make spreadsheet to then manually compare

    '''
    output_df = pd.DataFrame()
    output_df["Var A"] = col_1
    output_df["Var B"] = col_2
    output_df.to_excel("85% Collinear Pairs with Same Missingness.xlsx")
    '''

    return vars_to_drop


def calculate_missingness_avg(df):
    # Calculate missingness for each variable

    missingness = df.isna().mean(axis=1).groupby(level='Series Name').mean()

    return missingness


def calculate_missingness_ALL_years(df, year_columns):
    df_years = df.set_index(['Code', 'Series Name'])[year_columns]

    # Pivot the DataFrame
    df_pivot = df_years.pivot_table(index=['Series', 'Code'], columns='Year', values='Value')

    # Calculate the proportion of missing data for each series across all years
    missing_proportion = df_pivot.isna().mean(axis=1)

    # Determine which series are at least 90% complete for every year
    complete_series = missing_proportion.groupby(level='Series').apply(lambda x: (x < 0.1).all())
    return complete_series


def format_df(df):
    df_stacked = df.stack().reset_index()
    df_stacked.rename(columns={'level_2': 'Year', 0: 'Value'}, inplace=True)

    # Transform the 'Year' column to clean it up and extract the year only
    df_stacked['Year'] = df_stacked['Year'].str.extract('(\d{4})')[0]

    # Pivot the DataFrame to get each 'Series Name' as a column
    df_pivoted = df_stacked.pivot_table(index=['Code', 'Year'], columns='Series Name', values='Value')

    return df_pivoted


def add_vars_to_drop():

    '''
    after manually selecting between collinear vars with the same missingness (see test_collinearity),
    drop the extra variables
    '''

    df = pd.read_excel("85% Collinear Pairs with Selected Var.xlsx")

    # Extract sets of unique values
    var_a_set = set(df['Var A'].dropna())  # dropna to ignore missing values
    var_b_set = set(df['Var B'].dropna())
    selected_var_set = set(df['Var Selected'].dropna())

    # Calculate the difference
    result_set = (var_a_set.union(var_b_set)) - selected_var_set

    return result_set


def select_covariates(df, missingness_param, collinearity_param, start_year, end_year):
    # replace missing vals with NaN
    df = replace_missing_values(df)

    # Extract year columns based on the specified range
    year_columns = [f"{year} [YR{year}]" for year in range(start_year, end_year + 1)]
    df_years = df.set_index(['Code', 'Series Name'])[year_columns]

    missingness = calculate_missingness_avg(df_years)

    # Select variables with missingness less than the specified threshold
    selected_vars = missingness[missingness < missingness_param].index.tolist()

    # complete_series = calculate_missingness_ALL_years(df, year_columns)

    # Filter the dataframe to include only the selected variables
    df_selected = df[df['Series Name'].isin(selected_vars)]

    df_final = df_selected.set_index(['Code', 'Series Name'])[year_columns]

    df_pivoted = format_df(df_final)

    # Test for collinearity
    vars_to_drop = test_collinearity(df_pivoted, missingness, collinearity_param)

    # get list of variables that were manually selected to be dropped out of collinear pairs with same missingness
    extra_vars_to_drop = add_vars_to_drop()
    vars_to_drop = vars_to_drop.union(extra_vars_to_drop)

    # Final list of variables after removing collinear variables
    final_vars = [var for var in selected_vars if var not in vars_to_drop]

    always_included_vars = ["Gini index", "Government Effectiveness: Estimate"]
    for var in always_included_vars:
        if var in df['Series Name'].values and var not in final_vars:
            final_vars.append(var)

    print(len(final_vars))

    # Filter the original dataframe to include only the final variables
    df_res = df_selected[df_selected['Series Name'].isin(final_vars)]

    return df_res


def get_dataframe():
    df_1 = pd.read_excel('WDI/WDI_1.xlsx')
    df_2 = pd.read_excel('WDI/WDI_2.xlsx')
    df_3 = pd.read_excel('WDI/WDI_3.xlsx')
    df_4 = pd.read_excel('WDI/WDI_4.xlsx')
    df_5 = pd.read_excel('WDI/WDI_5.xlsx')
    df_6 = pd.read_excel('WDI/WDI_6.xlsx')
    df_7 = pd.read_excel('WDI/WDI_7.xlsx')

    dfs = [df_1, df_2, df_3, df_4, df_5, df_6, df_7]

    df_later_years = pd.concat(dfs, ignore_index=True)

    df_earlier_1 = pd.read_excel('WDI/1960-1973(1).xlsx')
    df_earlier_2 = pd.read_excel('WDI/1960-1973(2).xlsx')

    dfs_earlier = [df_earlier_1, df_earlier_2]
    df_earlier_years = pd.concat(dfs_earlier, ignore_index=True)

    merged_df = pd.merge(df_earlier_years, df_later_years, on=['Country Code', 'Series Name'], how='outer')

    return merged_df


def find_least_missing_covariates():
    merged_df = get_dataframe()

    # uncomment the following lines to save the full dataframe and/or read it in to skip merging step
    # merged_df.to_excel('WDI/all_indicators.xlsx')
    # merged_df = pd.read_excel("WDI/all_indicators.xlsx")

    # filter only large countries
    large_countries = pd.read_excel('Populations.xlsx')
    large_country_codes = list(large_countries["Code"])
    merged_df = merged_df[merged_df["Code"].isin(large_country_codes)]

    # set parameters
    missingness_param = 0.1
    collinearity_param = 0.85
    start_year = 2019
    end_year = 2019

    # get final filtered dataframe
    df_final = select_covariates(merged_df, missingness_param, collinearity_param, start_year, end_year)

    df_final.to_excel("WDI/covariates_list_85%_collinearity.xlsx")


def main():
    # test_missingness_individual_covariate()
    find_least_missing_covariates()


if __name__ == '__main__':
    main()
