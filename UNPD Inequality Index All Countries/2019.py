import pandas as pd
import statistics
import numpy as np
import math
from numpy import array, exp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

SAMPLE_SIZE = 100000.0

def func1(x, a, b, c):
    return a*x**2+b*x+c

def analyze(df):
    age = list(df["Age (x)"])
    deaths = df["Number of deaths d(x,n)"]
    print("deaths = ", deaths)
    print("age = ", age)
    p_AaD = list(deaths / SAMPLE_SIZE)
    print(p_AaD)

    under_5_mortality = p_death_under_5(p_AaD)

    mode = max(p_AaD)

    p_AaD_list = list(p_AaD)
    index = p_AaD_list.index(mode)  # find index of mode value

    AaD_mode = age[index]  # find age at that index
    print('modal age at death = :', AaD_mode)

    # option A
    if AaD_mode == 100:
        p_AaD_list[index] = 0
        new_mode = max(p_AaD_list)
        new_index = p_AaD_list.index(new_mode)
        AaD_mode = age[new_index]

    # option B
    #if AaD_mode == 100:
        #start = age.index(90)
        #end = age.index(100)
        #y = array(p_AaD_list[start:end])
       # x = array(range(len(y)))
        #params, covs = curve_fit(func1, x, y)
        #a, b, c = params[0], params[1], params[2]

       # start_new = end
        #end_new = end + 11
        #next_y = func1(start_new, a, b, c)
        #p_AaD_list[start_new] = next_y
        #for next_x in range(start_new+1, end_new):
          #  next_y = func1(next_x, a, b, c)
          #  p_AaD_list.append(next_y)
       # new_mode = max(p_AaD_list)
      #  new_index = p_AaD_list.index(new_mode)
       # AaD_mode = age[new_index]
        #print(AaD_mode)

    abs_dev = absolute_deviation(age, p_AaD, AaD_mode)
    standard_dev = standard_deviation(age, p_AaD, AaD_mode)

    dystopian_val_absdev = perfect_inequality_sim(AaD_mode, 'absolute')
    dystopian_val_stdev = perfect_inequality_sim(AaD_mode, 'standard')
    print("variance maximal: ", dystopian_val_stdev)
    print("variance actual: ", standard_dev)

    return abs_dev / dystopian_val_absdev, standard_dev / dystopian_val_stdev, AaD_mode, under_5_mortality

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

def p_death_under_5(p_AaD):
    deaths_under_5 = 0
    print(p_AaD)
    for i in range(5):
        deaths_under_5 += p_AaD[i]
    return deaths_under_5

def calculate_life_expectancy(df):
    age_and_expectation_cols = ['Age (x)', 'Expectation of life e(x)']
    life_expectancies = list(df[age_and_expectation_cols].sum(axis=1))
    return statistics.mean(life_expectancies)

def populate_variable(database, country_code, name):
    dataframe = database[database["Code"] == country_code]
    if dataframe.empty:
        return ''
    else:
        GDP = list(dataframe[name])[0]
        return GDP

def populate_democracy_variable(database, country):
    democracy_df = database[database["Country"].str.strip() == country.strip()]
    if democracy_df.empty:
        print('here')
        return ''
    democracy_val = list(democracy_df['democracy_polity'])[0]
    return democracy_val

def populate_region_variable(database, country):
    region_df = database[database["Country"].str.strip() == country.strip()]
    if region_df.empty:
        return ''
    else:
        region = list(region_df['Region'])[0]
        if region == 'Oceania':
            region = 'Western Pacific Region'
        return region

def average_gini_index(data, country_code):
    dataframe = data[data["Code"] == country_code]
    if dataframe.empty:
        return ''
    years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
             2015, 2016, 2017, 2018, 2019, 2020]

    sum = 0.0
    num_measurements = 0.0
    for year in years:
        curr_year = list(dataframe[year])
        if curr_year[0] == 0:
            continue
        else:
            sum += curr_year[0]
            num_measurements += 1
    if sum == 0:
        return ''
    else:
        return float(sum / num_measurements)


def main():
    dystopian_val_absdev = perfect_inequality_sim(0, 'absolute')
    dystopian_val_stdev = perfect_inequality_sim(0, 'standard')

    '''print("perfect inequality = ", dystopian_val_absdev)'''

    input_df = pd.read_excel('Input_Dataframes/Country_Data_2019.xlsx')

    GDP_data = pd.read_excel('GDP_2019.xlsx')
    gov_effectiveness_data = pd.read_excel('Gov_Effectiveness_Estimates_2019.xlsx')
    corruption_data = pd.read_excel('Corruption_2019.xlsx')
    fertility_rates_data = pd.read_excel('Fertility_Rates_2019.xlsx')
    gini_index_data = pd.read_excel('Gini Index Last 15 Years.xlsx')
    region_data = pd.read_excel('WHO_Region_Data.xlsx')
    income_class_data = pd.read_excel('Income_Sorting_Data_Numbers.xlsx')
    large_countries = pd.read_excel('Populations.xlsx')
    urbanization_data = pd.read_excel('Urbanization_2019.xlsx')
    stability_data = pd.read_excel('Political Stability_2019.xlsx')
    life_expectancy_data = pd.read_excel('Life Expectancy_2019.xlsx')
    democracy_all_years = pd.read_csv('democracy-polity.csv')
    democracy_data = democracy_all_years[democracy_all_years["Year"] == 2018]

    output_df = pd.DataFrame()

    countries = []
    country_codes = list(input_df["ISO3 Alpha-code"].unique())
    stdev_vals = []
    abs_dev_vals = []
    modal_ages = []
    u5_mortality_vals = []
    GDPs = []
    FRs = []
    corruption_vals = []
    income_classifications = []
    gov_effectiveness_vals = []
    populations = []
    regions = []
    gini_index_vals = []
    life_expectancies = []
    urbanization_rates = []
    stability_estimates = []
    democracy_vals = []

    large_countries_codes = list(large_countries["Code"])

    country_df = input_df[input_df["ISO3 Alpha-code"] == 'USA']
    abs_dev, standard_dev, modal_age, u5_mortality = analyze(country_df)
    print("USA index = ", standard_dev)

    for country_code in country_codes:
        if country_code not in large_countries_codes:
            continue

        country_df = input_df[input_df["ISO3 Alpha-code"] == country_code]
        abs_dev, standard_dev, modal_age, u5_mortality = analyze(country_df)
        stdev_vals.append(standard_dev*10**3)
        abs_dev_vals.append(abs_dev*10**3)
        modal_ages.append(modal_age)
        u5_mortality_vals.append(u5_mortality)
        country = list(country_df["Country"])[0]
        countries.append(country)
        print(country)

        democracy_vals.append(populate_variable(democracy_data, country_code, 'democracy_polity'))
        GDPs.append(populate_variable(GDP_data, country_code, 'GDP per capita'))
        FRs.append(populate_variable(fertility_rates_data, country_code, 'FR'))
        corruption_vals.append(populate_variable(corruption_data, country_code, 'Corruption'))
        gov_effectiveness_vals.append(populate_variable(gov_effectiveness_data, country_code, 'Value'))
        gini_index_vals.append(average_gini_index(gini_index_data, country_code))
        regions.append(populate_region_variable(region_data, country))
        populations.append(populate_variable(large_countries, country_code, 'Population'))
        income_classifications.append(populate_variable(income_class_data, country_code, 'Year'))
        life_expectancies.append(populate_variable(life_expectancy_data, country_code, 'Life Expectancy'))
        urbanization_rates.append(populate_variable(urbanization_data, country_code, 'Percent Urbanization'))
        stability_estimates.append(populate_variable(stability_data, country_code, 'Political Stability and Absence of Violence'))

    output_df["Country"] = countries
    output_df["Absolute Deviation Index Value"] = abs_dev_vals
    output_df["Standard Deviation Index Value"] = stdev_vals
    output_df["Modal Age"] = modal_ages
    output_df["Under-5 Mortality"] = u5_mortality_vals
    output_df["GDP per capita"] = GDPs
    output_df["World Bank Income Classification"] = income_classifications
    output_df["Population"] = populations
    output_df["Fertility Rate"] = FRs
    output_df["Corruption"] = corruption_vals
    output_df["Gov Effectiveness Estimate"] = gov_effectiveness_vals
    output_df["Gini Index"] = gini_index_vals
    output_df["WHO Region"] = regions
    output_df["Life Expectancy"] = life_expectancies
    output_df["Percent Population Living in Urban Areas"] = urbanization_rates
    output_df["Political Stability and Absence of Violence/Terrorism"] = stability_estimates
    output_df["Democracy Index"] = democracy_vals

    num_countries_with_gini = 0.0
    total_countries = 0.0
    for gini in gini_index_vals:
        total_countries += 1
        if gini == '':
            continue
        num_countries_with_gini += 1
    percent_countries_with_gini = float(num_countries_with_gini / total_countries)

    output_df1 = output_df.sort_values(by=['Standard Deviation Index Value'])
    output_df1.to_excel("Inequality Index Values 10-1.xlsx")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
