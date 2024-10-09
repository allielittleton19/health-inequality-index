import pandas as pd
import numpy as np


# Function to calculate life expectancy at birth
def le0(data):
    return

# Function to calculate Gini coefficient
def gini_fun(x, nax, ndx, ex):
    ex = ex.tolist()
    print("ex: ", ex)
    e = np.ones(len(x))
    D = np.outer(ndx, ndx)
    x_ = x + nax
    X_ = np.abs(np.outer(x_, e) - np.outer(e, x_))
    G = np.sum(D * X_) / (2 * ex[0])
    return G


# Function to calculate relative Gini coefficient
def r_gini_fun(data):
    return gini_fun(data['Age (x)'], data['Average number of years lived a(x,n)'],
                    data['Number of deaths d(x,n)'] / 100000, data['Expectation of life e(x)'])


# Function to calculate indicators for all country-years
def indicators(data):

    print("indicators")
    years = data['Year'].unique()
    country_codes = data['ISO3 Alpha-code'].unique()
    output_df = pd.DataFrame()

    countries = []
    ginis = []
    codes = []
    df_years = []

    large_countries = pd.read_excel('Populations.xlsx')
    large_countries_codes = list(large_countries["Code"])

    for year in years:
        print("in loop")
        year_df = data[data["Year"] == year]
        for code in country_codes:
            if code not in large_countries_codes:
                continue
            print("code = ", code)
            print("year = ", year)
            ctryr = year_df[(year_df['ISO3 Alpha-code'] == code)]
            print(ctryr)

            # e0 = le0(ctryr)
            ex = data['Expectation of life e(x)'].tolist()
            e0 = ex[0]
            giniA = r_gini_fun(ctryr)

            print(giniA)

            gini0 = giniA * e0
            country = ctryr['Country']

            countries.append(country)
            ginis.append(gini0)
            codes.append(code)
            df_years.append(year)

    output_df["Country"] = countries
    output_df["Code"] = codes
    output_df["Year"] = df_years
    output_df["Gini"] = ginis

    return output_df


def main():
    print("in main")

    # Combine data

    wpp5 = pd.read_excel('Input_Dataframes/Country_Data_1950-1985.xlsx')

    # Generate and store indicators
    data_ind = indicators(wpp5)

    # Save to excel
    data_ind.to_excel("Gini_vals_1950-2022.xlsx")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
