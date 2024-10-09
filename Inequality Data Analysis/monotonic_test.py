
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import statsmodels.api as sm


def test_monotonic(df, index):
    '''

    for each country:
        sort country_df by {index}


    '''
    codes = df["Code"].unique()
    monotonic_sums = []
    countries = []

    for code in codes:
        country_df = df[df['Code'] == code]
        country_df = country_df.sort_values(by=f'{index}')
        country_df['LE_diff'] = country_df['Life Expectancy'].diff()

        country_df['monotonic'] = (country_df['LE_diff'] <= 0).astype(int)

        country_df.loc[country_df['LE_diff'].isna(), 'monotonic'] = pd.NA

        print(country_df)

        average_monotonic = country_df['monotonic'].mean()

        monotonic_sums.append(average_monotonic)

        country = list(country_df["Country"])[0]

        countries.append(country)

    new_df = pd.DataFrame()
    new_df["Country"] = countries
    new_df[f'Sum Monotonic {index}'] = monotonic_sums

    return new_df


def main():

    '''
    jamison_df = pd.read_excel("Input_DFs/Jamison_Indices_1950-2022 with LE.xlsx")
    blfmm_df = pd.read_excel("Input_DFs/BLFMM by Country 1950-2020 + modal age.xlsx")
    gini_df = pd.read_excel("Input_DFs/Gini by Country 1950-2019.xlsx")

    jamison_blfmm = pd.merge(jamison_df, blfmm_df, on=['Code', 'Year'])

    df = pd.merge(jamison_blfmm, gini_df, on=['Code', 'Year'])

    df.to_excel("jamison_blfmm_gini_LE.xlsx")
    '''


    df = pd.read_excel("jamison_blfmm_gini_LE.xlsx")

    jamison_df = test_monotonic(df, 'Jamison Index')
    gini_df = test_monotonic(df, 'Gini')
    blfmm_df = test_monotonic(df, 'BLFMM')

    merged_df = jamison_df.merge(gini_df, on='Country').merge(blfmm_df, on='Country')

    merged_df.to_excel("Monotonic Test All Indices.xlsx")



if __name__ == '__main__':
    main()
