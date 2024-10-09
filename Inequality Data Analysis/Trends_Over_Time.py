import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import statsmodels.api as sm


def plot_each_country(df):

    codes = df["Code"].unique()

    # plot each country's BLFMM, gini, jamison, and modal age over the years
    for code in codes:
        print("in loop")

        subset = df[df['Code'] == code]
        country = list(subset["Country"])[0]

        plt.figure(figsize=(10, 6))
        plt.plot(subset['Year'], subset['Jamison Index'], label='Jamison', marker='o', color='b')
        plt.title(f'{country} Jamison Over Time')
        plt.xlabel('Year')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join('Index Plots', f'jamison_over_time_{country}.png'))

        plt.figure(figsize=(10, 6))
        plt.plot(subset['Year'], subset['BLFMM'], label='BLFMM', marker='o', color='g')
        plt.title(f'{country} BLFMM Over Time')
        plt.xlabel('Year')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join('Index Plots', f'blfmm_over_time_{country}.png'))

        plt.figure(figsize=(10, 6))
        plt.plot(subset['Year'], subset['Gini'], label='Gini', marker='o', color='g')
        plt.title(f'{country} Gini Over Time')
        plt.xlabel('Year')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join('Index Plots Gini', f'gini_over_time_{country}.png'))

        plt.figure(figsize=(10, 6))
        plt.plot(subset['Year'], subset['Modal Age'], label='Modal Age', marker='o', color='g')
        plt.title(f'{country} Modal Age at Death Over Time')
        plt.xlabel('Year')
        plt.ylabel('Modal Age at Death')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join('Modal Ages Over Time', f'modal_age_over_time_{country}.png'))


def non_parametric_fit(df, index, year):

    # plot non-parametric fit
    plt.figure(figsize=(10, 6))
    plt.scatter(df[f'{year}'], df[f'{index}'], label=f'{index}', color='b', marker='o')

    # Fit a non-parametric Lowess curve to the data
    lowess = sm.nonparametric.lowess(df[f'{index}'], df[f'{year}'], frac=0.3)

    # Extract the fitted values
    lowess_x = lowess[:, 0]
    lowess_y = lowess[:, 1]

    # Plot the Lowess fit
    plt.plot(lowess_x, lowess_y, label='Nonparametric Fit', color='r')

    plt.title(f'{index} by {year} for All Codes')
    plt.xlabel(f'{year}')
    plt.ylabel(f'{index}')
    plt.legend()
    plt.grid(True)

    plt.show()

def non_parametric_fit_bandwidth(df, index, year, bandwidth):

    # plot non-parametric fit
    plt.figure(figsize=(10, 6))
    plt.scatter(df[f'{year}'], df[f'{index}'], label=f'{index} (Within ±{bandwidth} years)', color='b', marker='o')

    # Fit a non-parametric Lowess curve to the data
    lowess = sm.nonparametric.lowess(df[f'{index}'], df[f'{year}'], frac=0.3)

    # Extract the fitted values
    lowess_x = lowess[:, 0]
    lowess_y = lowess[:, 1]

    # Plot the Lowess fit
    plt.plot(lowess_x, lowess_y, label='Nonparametric Fit', color='r')

    plt.title(f'{index} Values Over Time with Nonparametric Fit (Transition Year as Year 0, Bandwidth: ±{bandwidth} years)')
    plt.xlabel(f'{year}')
    plt.ylabel(f'{index}')
    plt.legend()
    plt.grid(True)

    plt.show()

def separate_lowess_fits(pre_df, post_df, index, year):
    plt.figure(figsize=(10, 6))

    # Scatter plot for pre-transition data
    plt.scatter(pre_df[year], pre_df[index], label=f'{index} (Transition Countries)', color='b', marker='o')

    # Fit a Lowess curve to the pre-transition data
    pre_lowess = sm.nonparametric.lowess(pre_df[index], pre_df[year], frac=0.3)
    pre_lowess_x = pre_lowess[:, 0]
    pre_lowess_y = pre_lowess[:, 1]

    # Plot the pre-transition Lowess fit
    plt.plot(pre_lowess_x, pre_lowess_y, label='Lowess Fit (Transition Countries)', color='r')

    # Scatter plot for post-transition data
    plt.scatter(post_df[year], post_df[index], label=f'{index} (Non-Transition Countries)', color='g', marker='o')

    # Fit a Lowess curve to the post-transition data
    post_lowess = sm.nonparametric.lowess(post_df[index], post_df[year], frac=0.3)
    post_lowess_x = post_lowess[:, 0]
    post_lowess_y = post_lowess[:, 1]

    # Plot the post-transition Lowess fit
    plt.plot(post_lowess_x, post_lowess_y, label='Lowess Fit (Non-Transition Countries)', color='m')

    plt.title(f'{index} Values Over Time with Separate Lowess Fits')
    plt.xlabel(f'{year}')
    plt.ylabel(f'{index}')
    plt.legend()
    plt.grid(True)

    plt.show()


def quadratic_fit(df, index, year):
    # plot quadratic fit

    plt.figure(figsize=(10, 6))
    plt.scatter(df[f'{year}'], df[f'{index}'], label=f'{index}', color='b', marker='o')

    # Fit a quadratic equation to the data
    coefficients = np.polyfit(df[f'{year}'], df[f'{index}'], 2)
    polynomial = np.poly1d(coefficients)

    # Generate values for the fitted curve
    years = np.linspace(df[f'{year}'].min(), df[f'{year}'].max(), 100)
    fitted_values = polynomial(years)

    # Plot the fitted curve
    plt.plot(years, fitted_values, label='Quadratic Fit', color='r')

    plt.title(f'{index} Values Over Time with Quadratic Fit (Transition Year as Year 0)')
    plt.xlabel(f'{year}')
    plt.ylabel(f'{index}')
    plt.legend()
    plt.grid(True)

    plt.show()


def separate_quadratic_fits(pre_df, post_df, index, year):
    plt.figure(figsize=(10, 6))

    # Scatter plot for pre-transition data
    plt.scatter(pre_df[year], pre_df[index], label=f'{index} Transition Countries', color='b', marker='o')

    # Fit a quadratic equation to the pre-transition data
    pre_coefficients = np.polyfit(pre_df[year], pre_df[index], 2)
    pre_polynomial = np.poly1d(pre_coefficients)

    # Generate values for the pre-transition fitted curve
    pre_years = np.linspace(pre_df[year].min(), pre_df[year].max(), 100)
    pre_fitted_values = pre_polynomial(pre_years)

    # Plot the pre-transition fitted curve
    plt.plot(pre_years, pre_fitted_values, label='Quadratic Fit (Transition)', color='r')

    # Scatter plot for post-transition data
    plt.scatter(post_df[year], post_df[index], label=f'{index} Non-Transition Countries', color='g', marker='o')

    # Fit a quadratic equation to the post-transition data
    post_coefficients = np.polyfit(post_df[year], post_df[index], 2)
    post_polynomial = np.poly1d(post_coefficients)

    # Generate values for the post-transition fitted curve
    post_years = np.linspace(post_df[year].min(), post_df[year].max(), 100)
    post_fitted_values = post_polynomial(post_years)

    # Plot the post-transition fitted curve
    plt.plot(post_years, post_fitted_values, label='Quadratic Fit (Non-Transition)', color='m')

    plt.title(f'{index} Values Over Time with Separate Quadratic Fits')
    plt.xlabel(f'{year}')
    plt.ylabel(f'{index}')
    plt.legend()
    plt.grid(True)

    plt.show()


def quadratic_fit_with_bandwidth(df, index, year, bandwidth):
    plt.figure(figsize=(10, 6))

    # Scatter plot for the data within the specified bandwidth
    plt.scatter(df[year], df[index], label=f'{index} (Bandwidth: ±{bandwidth} years)', color='b', marker='o')

    # Fit a quadratic equation to the data
    coefficients = np.polyfit(df[year], df[index], 2)
    polynomial = np.poly1d(coefficients)

    # Generate values for the fitted curve
    years = np.linspace(df[year].min(), df[year].max(), 100)
    fitted_values = polynomial(years)

    # Plot the fitted curve
    plt.plot(years, fitted_values, label='Quadratic Fit', color='r')

    plt.title(f'{index} Values Over Time with Quadratic Fit (Transition Year as Year 0, Bandwidth: ±{bandwidth} years)')
    plt.xlabel(f'{year}')
    plt.ylabel(f'{index}')
    plt.legend()
    plt.grid(True)

    plt.show()


def separate_quadratic_fit_with_bandwidth(pre_transition_df, post_transition_df, index, year, bandwidth):
    plt.scatter(pre_transition_df[year], pre_transition_df[index], label=f'{index} (Pre-Transition)', color='b',
                marker='o')

    # Fit a quadratic equation to the pre-transition data
    if not pre_transition_df.empty:
        pre_coefficients = np.polyfit(pre_transition_df[year], pre_transition_df[index], 2)
        pre_polynomial = np.poly1d(pre_coefficients)

        # Generate values for the pre-transition fitted curve
        pre_years = np.linspace(pre_transition_df[year].min(), pre_transition_df[year].max(), 100)
        pre_fitted_values = pre_polynomial(pre_years)

        # Plot the pre-transition fitted curve
        plt.plot(pre_years, pre_fitted_values, label='Quadratic Fit (Pre-Transition)', color='r')

    # Scatter plot for post-transition data
    plt.scatter(post_transition_df[year], post_transition_df[index], label=f'{index} (Post-Transition)', color='g',
                marker='o')

    # Fit a quadratic equation to the post-transition data
    if not post_transition_df.empty:
        post_coefficients = np.polyfit(post_transition_df[year], post_transition_df[index], 2)
        post_polynomial = np.poly1d(post_coefficients)

        # Generate values for the post-transition fitted curve
        post_years = np.linspace(post_transition_df[year].min(), post_transition_df[year].max(), 100)
        post_fitted_values = post_polynomial(post_years)

        # Plot the post-transition fitted curve
        plt.plot(post_years, post_fitted_values, label='Quadratic Fit (Post-Transition)', color='m')

    plt.title(f'{index} Values Over Time with Quadratic Fit (Transition Year as Year 0, Bandwidth: ±{bandwidth} years)')
    plt.xlabel(f'{year}')
    plt.ylabel(f'{index}')
    plt.legend()
    plt.grid(True)

    plt.show()


def separate_non_parametric_fit_bandwidth(pre_transition_df, post_transition_df, index, year,
                                      bandwidth):
    plt.figure(figsize=(10, 6))


    # Scatter plot for pre-transition data
    plt.scatter(pre_transition_df[year], pre_transition_df[index], label=f'{index} (Pre-Transition)', color='b',
                marker='o')

    # Fit a LOWESS curve to the pre-transition data
    if not pre_transition_df.empty:
        pre_lowess = sm.nonparametric.lowess(pre_transition_df[index], pre_transition_df[year], frac=0.3)
        pre_lowess_x = pre_lowess[:, 0]
        pre_lowess_y = pre_lowess[:, 1]

        # Plot the pre-transition LOWESS fit
        plt.plot(pre_lowess_x, pre_lowess_y, label='LOWESS Fit (Pre-Transition)', color='r')

    # Scatter plot for post-transition data
    plt.scatter(post_transition_df[year], post_transition_df[index], label=f'{index} (Post-Transition)', color='g',
                marker='o')

    # Fit a LOWESS curve to the post-transition data
    if not post_transition_df.empty:
        post_lowess = sm.nonparametric.lowess(post_transition_df[index], post_transition_df[year], frac=0.3)
        post_lowess_x = post_lowess[:, 0]
        post_lowess_y = post_lowess[:, 1]

        # Plot the post-transition LOWESS fit
        plt.plot(post_lowess_x, post_lowess_y, label='LOWESS Fit (Post-Transition)', color='m')

    plt.title(f'{index} Values Over Time with LOWESS Fit (Transition Year as Year 0, Bandwidth: ±{bandwidth} years)')
    plt.xlabel(f'{year}')
    plt.ylabel(f'{index}')
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_transition_years(df, index):

    # Sort the data by Country and Year
    # df = df.sort_values(by=['Country', 'Year'])

    # Create a shifted column to compare current and previous year's index values
    df['Prev_Modal_Age'] = df.groupby('Country')['Modal Age'].shift(1)

    df.to_excel("shifted_mode_check.xlsx")

    # Identify the transition year for each country where the index changes from 0 to a non-zero number
    transition_years = df[(df['Prev_Modal_Age'] == 0) & (df['Modal Age'] != 0)].groupby('Country')['Year'].first()

    # transition_years = df[(df['Prev_Modal_Age'] == 0) & (df['Modal Age'] != 0)][['Country', 'Year']]

    # Filter the original dataset to include only these countries
    filtered_df = df[df['Country'].isin(transition_years.index)]

    # Drop the Prev_Index column as it's no longer needed
    filtered_df = filtered_df.drop(columns=['Prev_Modal_Age'])

    non_transition_df = df[~df['Country'].isin(transition_years.index)]


    # Adjust the Year to have the transition year as year 0
    filtered_df['Adjusted_Year'] = filtered_df.apply(
        lambda row: row['Year'] - transition_years.loc[row['Country']], axis=1
    )



    # quadratic_fit(filtered_df, index, "Adjusted_Year")
    non_parametric_fit(filtered_df, index, "Adjusted_Year")
    non_parametric_fit(filtered_df, index, "Under-5 Mortality")
    non_parametric_fit(filtered_df, index, "Infant Mortality")

    # truncate life expectancy data
    filtered_df['Life Expectancy'] = filtered_df['Life Expectancy']
    filtered_df = filtered_df[filtered_df['Life Expectancy'] >= 30]

    non_parametric_fit(filtered_df, index, "Life Expectancy")

    # pre_transition_df = filtered_df[filtered_df['Adjusted_Year'] < 0]
    # post_transition_df = filtered_df[filtered_df['Adjusted_Year'] >= 0]

    # Plotting pre-transition and post-transition data on the same plot
    # separate_quadratic_fits(filtered_df, non_transition_df, index, 'Year')
    # separate_lowess_fits(filtered_df, non_transition_df, index, 'Year')

    '''

    bandwidths = [10, 20, 30, 40]

    # Plotting quadratic fit for each bandwidth
    for bandwidth in bandwidths:
        # Filter data within the specified bandwidth
        bandwidth_df = filtered_df[
            (filtered_df['Adjusted_Year'] >= -bandwidth) & (filtered_df['Adjusted_Year'] <= bandwidth)
            ]

        pre_transition_bandwidth_df = bandwidth_df[bandwidth_df['Adjusted_Year'] < 0]
        post_transition_bandwidth_df = bandwidth_df[bandwidth_df['Adjusted_Year'] >= 0]

        # separate_quadratic_fit_with_bandwidth(pre_transition_bandwidth_df, post_transition_bandwidth_df, index, 'Adjusted_Year', bandwidth)
        separate_non_parametric_fit_bandwidth(pre_transition_bandwidth_df, post_transition_bandwidth_df, index, 'Adjusted_Year', bandwidth)

        # non_parametric_fit_bandwidth(bandwidth_df, index, 'Adjusted_Year', bandwidth)
    '''

    '''
    # Plotting the data
    plt.figure(figsize=(12, 6))
    for country in filtered_df['Country'].unique():
        country_data = filtered_df[filtered_df['Country'] == country]
        plt.plot(country_data['Adjusted_Year'], country_data['BLFMM'], marker='o', label=country)

    plt.xlabel('Adjusted Year')
    plt.ylabel('BLFMM')
    plt.title('BLFMM Values for transition countries')
    plt.legend(title='Country')
    plt.grid(True)
    plt.show()
    '''

    '''
    # Create output directory for plots
    output_dir = 'Adjusted Year Plots'
    os.makedirs(output_dir, exist_ok=True)
    
    
    # Plotting the data
    for country in filtered_df['Country'].unique():
        country_data = filtered_df[filtered_df['Country'] == country]

        plt.figure(figsize=(12, 6))
        plt.plot(country_data['Adjusted_Year'], country_data['BLFMM'], marker='o', label=country)

        plt.xlabel('Adjusted Year')
        plt.ylabel('BLFMM')
        plt.title(f'BLFMM for {country} with Transition Year as Year 0')
        plt.legend(title='Country')
        plt.grid(True)

        # Save each plot as an image file
        plt.savefig(os.path.join(output_dir, f'{country}_transition_plot.png'))
        plt.close()

    print("Plots for each country have been saved in the 'Adjusted Year Plots' directory.")
    '''


def plot_indices_for_all_countries(df):

    # Sort the data by Country and Year
    df = df.sort_values(by=['Country', 'Year'])

    # Create a shifted column to compare current and previous year's index values
    df['Prev_Modal_Age'] = df.groupby('Country')['Modal Age'].shift(1)

    # Identify the transition year for each country where the index changes from 0 to a non-zero number
    transition_years = df[(df['Prev_Modal_Age'] == 0) & (df['Modal Age'] != 0)].groupby('Country')['Year'].first()

    # Filter the original dataset to include only these countries
    filtered_df = df[df['Country'].isin(transition_years.index)]

    # Drop the Prev_Index column as it's no longer needed
    filtered_df = filtered_df.drop(columns=['Prev_Modal_Age'])

    # Adjust the Year to have the transition year as year 0
    filtered_df['Adjusted_Year'] = filtered_df.apply(
        lambda row: row['Year'] - transition_years.loc[row['Country']], axis=1
    )

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Gini and Jamison Index on the left y-axis
    ax1.scatter(filtered_df['Adjusted_Year'], filtered_df['Gini'], color='b', marker='o', label='Gini')
    ax1.scatter(filtered_df['Adjusted_Year'], filtered_df['Jamison Index'], color='g', marker='o', label='Jamison Index')

    ax1.set_xlabel('Adjusted Year')
    ax1.set_ylabel('Gini / Jamison Index')
    ax1.legend(loc='upper left')

    # Plot BLFMM on the right y-axis
    ax2 = ax1.twinx()
    ax2.scatter(filtered_df['Adjusted_Year'], filtered_df['BLFMM'], color='r', marker='o', label='BLFMM')
    ax2.set_ylabel('BLFMM')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.title('Indices Over Time (Transition Year as Year 0)')
    plt.grid(True)
    plt.show()


def plot_indices_with_quadratic_fits(df):

    # Sort the data by Country and Year
    df = df.sort_values(by=['Country', 'Year'])

    # Create a shifted column to compare current and previous year's index values
    df['Prev_Modal_Age'] = df.groupby('Country')['Modal Age'].shift(1)

    # Identify the transition year for each country where the index changes from 0 to a non-zero number
    transition_years = df[(df['Prev_Modal_Age'] == 0) & (df['Modal Age'] != 0)].groupby('Country')['Year'].first()

    # Filter the original dataset to include only these countries
    filtered_df = df[df['Country'].isin(transition_years.index)]

    # Drop the Prev_Index column as it's no longer needed
    filtered_df = filtered_df.drop(columns=['Prev_Modal_Age'])

    # Adjust the Year to have the transition year as year 0
    filtered_df['Adjusted_Year'] = filtered_df.apply(
        lambda row: row['Year'] - transition_years.loc[row['Country']], axis=1
    )

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Separate the data into pre-transition and post-transition
    pre_transition_df = filtered_df[filtered_df['Adjusted_Year'] < 0]
    post_transition_df = filtered_df[filtered_df['Adjusted_Year'] >= 0]

    # Plot Gini and Jamison Index on the left y-axis
    ax1.scatter(filtered_df['Adjusted_Year'], filtered_df['Gini'], color='b', marker='o', label='Gini Points')
    ax1.scatter(filtered_df['Adjusted_Year'], filtered_df['Jamison Index'], color='g', marker='o', label='Jamison Index Points')

    ax1.set_xlabel('Adjusted Year')
    ax1.set_ylabel('Gini / Jamison Index')
    ax1.legend(loc='upper left')

    # Fit and plot quadratic curve for pre-transition Gini
    if not pre_transition_df.empty:
        gini_pre_coeff = np.polyfit(pre_transition_df['Adjusted_Year'], pre_transition_df['Gini'], 2)
        gini_pre_poly = np.poly1d(gini_pre_coeff)
        gini_pre_x = np.linspace(pre_transition_df['Adjusted_Year'].min(), pre_transition_df['Adjusted_Year'].max(), 100)
        gini_pre_y = gini_pre_poly(gini_pre_x)
        ax1.plot(gini_pre_x, gini_pre_y, color='cyan', linestyle='--', linewidth=2, label='Gini (Pre) Fit')

        jamison_pre_coeff = np.polyfit(pre_transition_df['Adjusted_Year'], pre_transition_df['Jamison Index'], 2)
        jamison_pre_poly = np.poly1d(jamison_pre_coeff)
        jamison_pre_x = np.linspace(pre_transition_df['Adjusted_Year'].min(), pre_transition_df['Adjusted_Year'].max(), 100)
        jamison_pre_y = jamison_pre_poly(jamison_pre_x)
        ax1.plot(jamison_pre_x, jamison_pre_y, color='lime', linestyle='--', linewidth=2, label='Jamison Index (Pre) Fit')

    # Fit and plot quadratic curve for post-transition Gini
    if not post_transition_df.empty:
        gini_post_coeff = np.polyfit(post_transition_df['Adjusted_Year'], post_transition_df['Gini'], 2)
        gini_post_poly = np.poly1d(gini_post_coeff)
        gini_post_x = np.linspace(post_transition_df['Adjusted_Year'].min(), post_transition_df['Adjusted_Year'].max(), 100)
        gini_post_y = gini_post_poly(gini_post_x)
        ax1.plot(gini_post_x, gini_post_y, color='darkblue', linestyle='-', linewidth=2, label='Gini (Post) Fit')

        jamison_post_coeff = np.polyfit(post_transition_df['Adjusted_Year'], post_transition_df['Jamison Index'], 2)
        jamison_post_poly = np.poly1d(jamison_post_coeff)
        jamison_post_x = np.linspace(post_transition_df['Adjusted_Year'].min(), post_transition_df['Adjusted_Year'].max(), 100)
        jamison_post_y = jamison_post_poly(jamison_post_x)
        ax1.plot(jamison_post_x, jamison_post_y, color='darkgreen', linestyle='-', linewidth=2, label='Jamison Index (Post) Fit')

    # Plot BLFMM on the right y-axis
    ax2 = ax1.twinx()
    ax2.scatter(filtered_df['Adjusted_Year'], filtered_df['BLFMM'], color='r', marker='o', label='BLFMM Points')
    ax2.set_ylabel('BLFMM')
    ax2.legend(loc='upper right')

    # Fit and plot quadratic curve for pre-transition BLFMM
    if not pre_transition_df.empty:
        blfmm_pre_coeff = np.polyfit(pre_transition_df['Adjusted_Year'], pre_transition_df['BLFMM'], 2)
        blfmm_pre_poly = np.poly1d(blfmm_pre_coeff)
        blfmm_pre_x = np.linspace(pre_transition_df['Adjusted_Year'].min(), pre_transition_df['Adjusted_Year'].max(), 100)
        blfmm_pre_y = blfmm_pre_poly(blfmm_pre_x)
        ax2.plot(blfmm_pre_x, blfmm_pre_y, color='magenta', linestyle='--', linewidth=2, label='BLFMM (Pre) Fit')

    # Fit and plot quadratic curve for post-transition BLFMM
    if not post_transition_df.empty:
        blfmm_post_coeff = np.polyfit(post_transition_df['Adjusted_Year'], post_transition_df['BLFMM'], 2)
        blfmm_post_poly = np.poly1d(blfmm_post_coeff)
        blfmm_post_x = np.linspace(post_transition_df['Adjusted_Year'].min(), post_transition_df['Adjusted_Year'].max(), 100)
        blfmm_post_y = blfmm_post_poly(blfmm_post_x)
        ax2.plot(blfmm_post_x, blfmm_post_y, color='darkred', linestyle='-', linewidth=2, label='BLFMM (Post) Fit')

    fig.tight_layout()
    plt.title('Indices Over Time (Transition Year as Year 0) with Quadratic Fits')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.grid(True)
    plt.show()


def plot_indices_with_non_parametric_fits(df):

    # Sort the data by Country and Year
    df = df.sort_values(by=['Country', 'Year'])

    # Create a shifted column to compare current and previous year's index values
    df['Prev_Modal_Age'] = df.groupby('Country')['Modal Age'].shift(1)

    # Identify the transition year for each country where the index changes from 0 to a non-zero number
    transition_years = df[(df['Prev_Modal_Age'] == 0) & (df['Modal Age'] != 0)].groupby('Country')['Year'].first()

    # Filter the original dataset to include only these countries
    filtered_df = df[df['Country'].isin(transition_years.index)]

    # Drop the Prev_Index column as it's no longer needed
    filtered_df = filtered_df.drop(columns=['Prev_Modal_Age'])

    # Adjust the Year to have the transition year as year 0
    filtered_df['Adjusted_Year'] = filtered_df.apply(
        lambda row: row['Year'] - transition_years.loc[row['Country']], axis=1
    )

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Separate the data into pre-transition and post-transition
    pre_transition_df = filtered_df[filtered_df['Adjusted_Year'] < 0]
    post_transition_df = filtered_df[filtered_df['Adjusted_Year'] >= 0]

    # Plot Gini and Jamison Index on the left y-axis
    ax1.scatter(filtered_df['Adjusted_Year'], filtered_df['Gini'], color='b', marker='o', label='Gini Points')
    ax1.scatter(filtered_df['Adjusted_Year'], filtered_df['Jamison Index'], color='g', marker='o', label='Jamison Index Points')

    ax1.set_xlabel('Adjusted Year')
    ax1.set_ylabel('Gini / Jamison Index')
    ax1.legend(loc='upper left')

    # Fit and plot non-parametric curve for pre-transition Gini
    if not pre_transition_df.empty:
        gini_pre_lowess = sm.nonparametric.lowess(pre_transition_df['Gini'], pre_transition_df['Adjusted_Year'], frac=0.3)
        ax1.plot(gini_pre_lowess[:, 0], gini_pre_lowess[:, 1], color='cyan', linestyle='--', linewidth=2, label='Gini (Pre) Fit')

        jamison_pre_lowess = sm.nonparametric.lowess(pre_transition_df['Jamison Index'], pre_transition_df['Adjusted_Year'], frac=0.3)
        ax1.plot(jamison_pre_lowess[:, 0], jamison_pre_lowess[:, 1], color='lime', linestyle='--', linewidth=2, label='Jamison Index (Pre) Fit')

    # Fit and plot non-parametric curve for post-transition Gini
    if not post_transition_df.empty:
        gini_post_lowess = sm.nonparametric.lowess(post_transition_df['Gini'], post_transition_df['Adjusted_Year'], frac=0.3)
        ax1.plot(gini_post_lowess[:, 0], gini_post_lowess[:, 1], color='darkblue', linestyle='-', linewidth=2, label='Gini (Post) Fit')

        jamison_post_lowess = sm.nonparametric.lowess(post_transition_df['Jamison Index'], post_transition_df['Adjusted_Year'], frac=0.3)
        ax1.plot(jamison_post_lowess[:, 0], jamison_post_lowess[:, 1], color='darkgreen', linestyle='-', linewidth=2, label='Jamison Index (Post) Fit')

    # Plot BLFMM on the right y-axis
    ax2 = ax1.twinx()
    ax2.scatter(filtered_df['Adjusted_Year'], filtered_df['BLFMM'], color='r', marker='o', label='BLFMM Points')
    ax2.set_ylabel('BLFMM')
    ax2.legend(loc='upper right')

    # Fit and plot non-parametric curve for pre-transition BLFMM
    if not pre_transition_df.empty:
        blfmm_pre_lowess = sm.nonparametric.lowess(pre_transition_df['BLFMM'], pre_transition_df['Adjusted_Year'], frac=0.3)
        ax2.plot(blfmm_pre_lowess[:, 0], blfmm_pre_lowess[:, 1], color='magenta', linestyle='--', linewidth=2, label='BLFMM (Pre) Fit')

    # Fit and plot non-parametric curve for post-transition BLFMM
    if not post_transition_df.empty:
        blfmm_post_lowess = sm.nonparametric.lowess(post_transition_df['BLFMM'], post_transition_df['Adjusted_Year'], frac=0.3)
        ax2.plot(blfmm_post_lowess[:, 0], blfmm_post_lowess[:, 1], color='darkred', linestyle='-', linewidth=2, label='BLFMM (Post) Fit')

    fig.tight_layout()
    plt.title('Indices Over Time (Transition Year as Year 0) with Non-Parametric Fits')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.grid(True)
    plt.show()


def plot_quadratic_fit(x, y, ax, color, label):
    coeffs = np.polyfit(x, y, 2)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = poly(x_fit)
    ax.plot(x_fit, y_fit, color=color, label=label)


def GDP_plots(df, index, countries_subset, gdp_source):
    GDPs = pd.read_csv(f"gdp-per-capita {gdp_source}.csv")
    merged_df = pd.merge(df, GDPs, on=['Code', 'Year'])

    # Plot without lines connecting the points
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot BLFMM on the primary y-axis
    ax1.set_xlabel('Year')
    ax1.set_ylabel(f'{index}', color='tab:blue')
    ax1.scatter(merged_df['Year'], merged_df[f'{index}'], color='tab:blue', marker='o')
    plot_quadratic_fit(merged_df['Year'], merged_df[f'{index}'], ax1, 'tab:purple', f'Quadratic Fit {index}')

    # Create a secondary y-axis for GDP Per Capita
    ax2 = ax1.twinx()
    ax2.set_ylabel(f'{gdp_source} GDP Per Capita', color='tab:green')
    ax2.scatter(merged_df['Year'], merged_df['GDP Per Capita'], color='tab:green', marker='x')
    plot_quadratic_fit(merged_df['Year'], merged_df['GDP Per Capita'], ax2, 'tab:red', 'Quadratic Fit GDP Per Capita')

    fig.tight_layout()
    plt.title(f'{gdp_source} GDP Per Capita and {index} Trends Over Time')
    plt.grid(True)

    directory = f'{gdp_source} GDP {index} Plots'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, f'{index} and {gdp_source} GDP {countries_subset} countries.png')
    plt.savefig(filename)
    plt.show()

    '''

    directory = f'{gdp_source} GDP {index} Plots Each Country'
    if not os.path.exists(directory):
        os.makedirs(directory)
    countries = merged_df['Country'].unique()

    for country in countries:
        country_data = merged_df[merged_df['Country'] == country]

        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.set_title(f'GDP Per Capita and {index} Trends Over Time for {country}')
        ax1.set_xlabel('Year')
        ax1.set_ylabel(f'{index}', color='tab:blue')
        ax1.scatter(country_data['Year'], country_data[f'{index}'], color='tab:blue', marker='o', linestyle='-')

        ax2 = ax1.twinx()
        ax2.set_ylabel('GDP Per Capita', color='tab:green')
        ax2.scatter(country_data['Year'], country_data['GDP Per Capita'], color='tab:green', marker='x', linestyle='-')

        fig.tight_layout()
        ax1.grid(True)
        ax1.legend(loc='upper left')

        # Save the plot with the specified filename in the new directory
        filename = os.path.join(directory, f'{country}_GDP_and_{index}_trends.png')
        plt.savefig(filename)

    '''

def main():

    '''
    jamison_df = pd.read_excel("Input_DFs/Jamison_Indices_1950-2021(1).xlsx")
    blfmm_df = pd.read_excel("Input_DFs/BLFMM by Country 1950-2020 + modal age + mortality.xlsx")
    gini_df = pd.read_excel("Input_DFs/Gini by Country 1950-2019.xlsx")

    jamison_blfmm = pd.merge(jamison_df, blfmm_df, on=['Code', 'Year'])

    df = pd.merge(jamison_blfmm, gini_df, on=['Code', 'Year'])

    df.to_excel("jamison_blfmm_gini.xlsx")
    '''

    ''''''

    df = pd.read_excel("jamison_blfmm_gini.xlsx")

    # plot_transition_years(df, "BLFMM")
    plot_transition_years(df, "Gini")
    plot_transition_years(df, "Jamison Index")

    '''
    # plot_indices_for_all_countries(df)

    # plot_indices_with_quadratic_fits(df)


    GDP_plots(df, 'BLFMM', 'all', 'Maddison Project')
    # GDP_plots(df, 'Jamison Index', 'all', 'Maddison Project')
    # GDP_plots(df, 'Gini', 'all', 'Maddison Project')


    # GDP_plots(df, 'BLFMM', 'all', 'World Bank')
    # GDP_plots(df, 'Jamison Index', 'all', 'World Bank')
    # GDP_plots(df, 'Gini', 'all', 'World Bank')

    # plot_indices_with_non_parametric_fits(df)
    df['Prev_Modal_Age'] = df.groupby('Country')['Modal Age'].shift(1)

    # Identify the transition year for each country where the index changes from 0 to a non-zero number
    transition_years = df[(df['Prev_Modal_Age'] == 0) & (df['Modal Age'] != 0)].groupby('Country')['Year'].first()

    # Filter the original dataset to include only these countries
    filtered_df = df[df['Country'].isin(transition_years.index)]


    #GDP_plots(filtered_df, 'BLFMM', 'transition', 'Maddison Project')
    #GDP_plots(filtered_df, 'Jamison Index', 'transition', 'Maddison Project')
    #GDP_plots(filtered_df, 'Gini', 'transition', 'Maddison Project')


    #GDP_plots(filtered_df, 'BLFMM', 'transition', 'World Bank')
    #GDP_plots(filtered_df, 'Jamison Index', 'transition', 'World Bank')
    #GDP_plots(filtered_df, 'Gini', 'transition', 'World Bank')

    '''

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
