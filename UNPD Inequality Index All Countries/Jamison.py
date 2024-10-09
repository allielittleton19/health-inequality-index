import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad


def create_survival_curve(df):

    INITIAL_POP = 100000

    # Calculate the survival probability Sx
    df['S(x)'] = df['Number of survivors l(x)'] / INITIAL_POP

    # Create interpolation function
    interpolator = interp1d(df['Age (x)'], df['S(x)'], kind='cubic')

    return interpolator


def calculate_index(life_expectancy, survival_function, df):
    max_age = df['Age (x)'].max()
    result = quad(survival_function, life_expectancy, max_age)

    numerator = result[0]

    j_index = numerator / life_expectancy

    return j_index


def square_survival_curve(age_range, life_expectancy):
    square_curve = np.ones_like(age_range)
    square_curve[age_range >= life_expectancy] = 0
    return square_curve


def plot_curve(country, interpolator, df, j_index, life_expectancy):
    # Generate a range of ages for smooth plotting
    age_range = np.linspace(df['Age (x)'].min(), df['Age (x)'].max(), 300)

    # Interpolated survival probabilities
    survival_probabilities = interpolator(age_range)

    square_curve = square_survival_curve(age_range, life_expectancy)

    # Plot the continuous survival curve
    plt.figure(figsize=(10, 6))
    plt.plot(age_range, survival_probabilities, label='Interpolated Survival Curve')
    plt.plot(age_range, square_curve, label='Square Survival Curve', linestyle='--')
    plt.xlabel('Age')
    plt.ylabel('Survival Probability')
    plt.title(f'{country} Survival Curve, r(s) = {j_index}')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():

    # input_df = pd.read_excel('Input_Dataframes/Country_Data_2019.xlsx')

    # country_codes = list(input_df["ISO3 Alpha-code"].unique())

    large_countries = pd.read_excel('Populations.xlsx')

    large_countries_codes = list(large_countries["Code"])

    '''
    #USA specific
    USA_df = input_df[input_df["ISO3 Alpha-code"] == 'USA']
    life_expectancy = USA_df.loc[USA_df['Age (x)'] == 0, 'Expectation of life e(x)'].values[0]
    survival_func = create_survival_curve(USA_df)
    j_index = calculate_index(life_expectancy, survival_func, USA_df)
    plot_curve('USA', survival_func, USA_df, j_index, life_expectancy)
    '''

    j_indices = []
    countries = []
    codes = []

    df1 = pd.read_excel('Input_Dataframes/Country_Data_1950-1985.xlsx')

    df2 = pd.read_excel('Input_Dataframes/Country_Data_1986-2019.xlsx')

    input_df = pd.concat([df1, df2], ignore_index=True)

    output_df = pd.DataFrame()
    life_expectancies = []
    countries = []
    country_codes = list(input_df["ISO3 Alpha-code"].unique())

    years = list(input_df["Year"].unique())
    df_years = []
    for year in years:
        year_df = input_df[input_df["Year"] == year]

        for country_code in country_codes:
            if country_code not in large_countries_codes:
                continue

            country_df = year_df[year_df["ISO3 Alpha-code"] == country_code]
            life_expectancy = country_df.loc[country_df['Age (x)'] == 0, 'Expectation of life e(x)'].values[0]
            survival_func = create_survival_curve(country_df)
            j_index = calculate_index(life_expectancy, survival_func, country_df)
            j_indices.append(j_index)
            country = list(country_df["Country"])[0]
            countries.append(country)
            df_years.append(list(country_df["Year"])[0])
            codes.append(country_code)
            life_expectancies.append(life_expectancy)

    output_df = pd.DataFrame()
    output_df["Year"] = df_years
    output_df["Country"] = countries
    output_df["Code"] = codes
    output_df["Life Expectancy"] = life_expectancies
    output_df["Jamison Index"] = j_indices

    output_df.to_excel("Jamison_Indices_1950-2022 with LE.xlsx")


if __name__ == '__main__':
    main()