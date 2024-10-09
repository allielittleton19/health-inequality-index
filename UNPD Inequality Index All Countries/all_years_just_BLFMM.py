import pandas as pd
import statistics
import numpy as np
import math

SAMPLE_SIZE = 100000.0


def analyze(df):
    age = list(df["Age (x)"])
    deaths = df["Number of deaths d(x,n)"]
    p_AaD = list(deaths / SAMPLE_SIZE)

    under_5_mortality = p_death_under_age(p_AaD, 5)
    infant_mortality = p_death_under_age(p_AaD, 1)

    mode = max(p_AaD)

    p_AaD_list = list(p_AaD)
    index = p_AaD_list.index(mode)  # find index of mode value

    AaD_mode = age[index]  # find age at that index

    # option A
    if AaD_mode == 100:
        p_AaD_list[index] = 0
        new_mode = max(p_AaD_list)
        new_index = p_AaD_list.index(new_mode)
        AaD_mode = age[new_index]

    # option B
    # if AaD_mode == 100:
    # start = age.index(90)
    # end = age.index(100)
    # y = array(p_AaD_list[start:end])
    # x = array(range(len(y)))
    # params, covs = curve_fit(func1, x, y)
    # a, b, c = params[0], params[1], params[2]

    # start_new = end
    # end_new = end + 11
    # next_y = func1(start_new, a, b, c)
    # p_AaD_list[start_new] = next_y
    # for next_x in range(start_new+1, end_new):
    #  next_y = func1(next_x, a, b, c)
    #  p_AaD_list.append(next_y)
    # new_mode = max(p_AaD_list)
    #  new_index = p_AaD_list.index(new_mode)
    # AaD_mode = age[new_index]
    # print(AaD_mode)

    # abs_dev = absolute_deviation(age, p_AaD, AaD_mode)

    standard_dev = standard_deviation(age, p_AaD, AaD_mode)

    dystopian_val_stdev = perfect_inequality_sim(AaD_mode, 'standard')

    standard_dev = standard_deviation(age, p_AaD, AaD_mode)

    return standard_dev / dystopian_val_stdev, AaD_mode, under_5_mortality, infant_mortality


# Our idea
def absolute_deviation(AaD_list, p_AaD_list, AaD_m):
    sum = 0.0
    n = len(AaD_list)
    for i in range(n):
        AaD = AaD_list[i]
        p_AaD = p_AaD_list[i]  # probability of death at current age
        sum = sum + (abs(AaD - AaD_m)*p_AaD)
    return sum


# Standard deviation
def standard_deviation(AaD_list, p_AaD_list, AaD_m):
    sum = 0.0
    n = len(AaD_list)
    for i in range(n):
        AaD = AaD_list[i]
        distance_sq = (AaD - AaD_m) * (AaD - AaD_m)  # (x_i - x_m)^2
        sum = sum + (distance_sq * p_AaD_list[i])
    return math.sqrt(sum)


# mode = 2%, all other vals = (98/99)%
def perfect_inequality_sim(mode, calc_method):
    sum = 0.0
    for AaD in range(101):
        p_AaD = float(98/99)
        if AaD == mode:
            p_AaD = .02
        if calc_method == 'absolute':
            sum = sum + (abs(AaD - mode) * p_AaD)
        else:
            distance_sq = (AaD - mode) * (AaD - mode)  # (x_i - x_m)^2
            sum = sum + math.sqrt(distance_sq * p_AaD)
    return sum


def p_death_under_age(p_AaD, age):
    deaths_under_age = 0
    print(p_AaD)
    for i in range(age):
        deaths_under_age += p_AaD[i]
    return deaths_under_age


def populate_variable(database, country_code, name):
    dataframe = database[database["Code"] == country_code]
    if dataframe.empty:
        return ''
    else:
        GDP = list(dataframe[name])[0]
        return GDP

def main():

    df1 = pd.read_excel('Input_Dataframes/Country_Data_1950-1985.xlsx')

    df2 = pd.read_excel('Input_Dataframes/Country_Data_1986-2019.xlsx')

    input_df = pd.concat([df1, df2], ignore_index=True)

    output_df = pd.DataFrame()

    countries = []
    country_codes = list(input_df["ISO3 Alpha-code"].unique())

    large_countries = pd.read_excel('Populations.xlsx')

    large_countries_codes = list(large_countries["Code"])

    codes = []
    stdev_vals = []
    modal_ages = []
    life_expectancies = []
    u5_mortalities = []
    infant_mortalities = []

    years = list(input_df["Year"].unique())
    df_years = []
    for year in years:
        print("in loop")
        year_df = input_df[input_df["Year"] == year]

        for country_code in country_codes:
            if country_code not in large_countries_codes:
                continue

            country_df = year_df[year_df["ISO3 Alpha-code"] == country_code]
            print(country_df)
            standard_dev, modal_age, u5_mortality, infant_mortality = analyze(country_df)
            stdev_vals.append(standard_dev*10**3)

            df_years.append(list(country_df["Year"])[0])
            countries.append(list(country_df["Country"])[0])

            life_expectancies.append(list(country_df["Expectation of life e(x)"])[0])

            u5_mortalities.append(u5_mortality)
            infant_mortalities.append(infant_mortality)

            codes.append(country_code)
            modal_ages.append(modal_age)

    output_df["Year"] = df_years
    output_df["Country"] = countries
    output_df["Code"] = codes
    output_df['Modal Age'] = modal_ages

    output_df["BLFMM"] = stdev_vals

    output_df["Life Expectancy"] = life_expectancies
    output_df["Infant Mortality"] = infant_mortalities
    output_df["Under-5 Mortality"] = u5_mortalities

    output_df2 = output_df.sort_values(by=['BLFMM'])
    output_df2.to_excel("BLFMM by Country 1950-2020 + modal age + mortality.xlsx")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
