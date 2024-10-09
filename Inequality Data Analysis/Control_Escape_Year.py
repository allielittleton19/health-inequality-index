
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess


def has_single_transition(df):

    df = df.sort_values(by=['Year'])
    df['Prev_Modal_Age'] = df.groupby('Country')['Modal Age'].shift(1)

    transition_counts = df[(df['Prev_Modal_Age'] == 0) & (df['Modal Age'] != 0)].groupby('Country')['Year'].count()
    single_transition_countries = transition_counts[transition_counts == 1].index

    print(single_transition_countries)
    filtered_df = df[df['Country'].isin(single_transition_countries)]

    transition_years = filtered_df[(filtered_df['Prev_Modal_Age'] == 0) & (filtered_df['Modal Age'] != 0)].groupby('Country')['Year'].min()

    filtered_df = filtered_df.drop(columns=['Prev_Modal_Age'])

    # Adjust the year to make the transition year equal to 1
    filtered_df['relative_year'] = filtered_df.apply(
        lambda row: row['Year'] - transition_years.loc[row['Country']] + 1, axis=1
    )

    return filtered_df


def plot_fitted_curve(filtered_df, model):
    # Extract coefficients
    coef = model.params

    # Calculate predicted values
    filtered_df['predicted_BLFMM'] = (
        coef['Intercept'] +
        coef['relative_year'] * filtered_df['relative_year'] +
        coef['relative_year_gt_0'] * filtered_df['relative_year_gt_0'] +
        coef['interaction_term'] * filtered_df['interaction_term'] +
        coef['relative_year_sq'] * filtered_df['relative_year_sq'] +
        coef['interaction_sq'] * filtered_df['interaction_sq']
    )

    # Plot the original data
    plt.scatter(filtered_df['relative_year'], filtered_df['BLFMM'], color='black', label='Original Data')

    # Plot the fitted quadratic curve
    plt.plot(filtered_df['relative_year'], filtered_df['predicted_BLFMM'], color='blue', label='Fitted Curve')

    plt.xlabel('Relative Year')
    plt.ylabel('BLFMM')
    plt.legend()
    plt.show()


def run_regression(filtered_df):
    filtered_df['relative_year_gt_0'] = filtered_df['relative_year'] > 0
    filtered_df['interaction_term'] = filtered_df['relative_year_gt_0'] * filtered_df['relative_year']

    # Convert boolean to integer for regression
    filtered_df['relative_year_gt_0'] = filtered_df['relative_year_gt_0'].astype(int)

    filtered_df['relative_year_sq'] = filtered_df['relative_year'] ** 2
    filtered_df['interaction_sq'] = filtered_df['relative_year_sq'] * filtered_df['relative_year_gt_0']

    filtered_df.to_excel("regression_terms(1).xlsx")

    print(filtered_df.columns)

    filtered_df['Year'] = filtered_df['Year'].astype('category')
    filtered_df['Country'] = filtered_df['Country'].astype('category')

    # Prepare the formula for the regression
    formula = 'BLFMM ~ relative_year + C(Country) + C(Year) + relative_year_gt_0 + interaction_term'
    # formula = 'BLFMM ~ relative_year + relative_year_sq + interaction_sq + C(Country) + C(Year) + relative_year_gt_0 + interaction_term'

    # Run the regression
    model = smf.ols(formula=formula, data=filtered_df).fit()

    summary_df = pd.DataFrame(model.summary().tables[1].data[1:], columns=model.summary().tables[1].data[0])
    summary_df.to_excel("regression_summary.xlsx", index=False)


    # get sample mean values for
    derived_means = filtered_df[['relative_year_gt_0']].mean()

    '''
    # This involves filtering the parameters for country-related coefficients
    country_effects = model.params.filter(like='C(Country)')
    year_effects = model.params.filter(like='C(Year)')

    # Calculate the mean of these country and year fixed effects
    mean_country_effect = country_effects.mean()
    mean_year_effect = year_effects.mean()
    derived_means['Year'] = mean_year_effect
    derived_means['Country'] = mean_country_effect
    '''

   # print(derived_means['relative_year_gt_0'])

    predict_df = pd.DataFrame({
        'relative_year': filtered_df['relative_year'],
        'interaction_term': filtered_df['interaction_term'],
        'relative_year_gt_0': derived_means['relative_year_gt_0'],
        'Country': filtered_df['Country'],
        'Year': filtered_df['Year'],
      #  'interaction_sq': filtered_df['interaction_sq'],
       # 'relative_year_sq': filtered_df['relative_year_sq']

    })

    predictions = model.predict(predict_df)

    predict_df['BLFMM_pred'] = predictions




    '''
    predict_df = filtered_df.drop(columns=['BLFMM'])
    predictions = model.predict(predict_df)
    predict_df['BLFMM_pred'] = predictions
    '''
    lowess_results_year = lowess(predict_df['BLFMM_pred'], filtered_df['relative_year'], frac=0.3)
    lowess_results_u5 = lowess(predict_df['BLFMM_pred'], filtered_df['Under-5 Mortality'], frac=0.3)
    lowess_results_infant = lowess(predict_df['BLFMM_pred'], filtered_df['Infant Mortality'], frac=0.3)

    # Plotting the predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(predict_df['relative_year'], predict_df['BLFMM_pred'], label='Predicted BLFMM', color='red')
    plt.plot(lowess_results_year[:, 0], lowess_results_year[:, 1], 'g-', label='Lowess Fit', linewidth=2)
    plt.xlabel('Relative Year')
    plt.ylabel('Predicted BLFMM')
    plt.title('Predicted BLFMM by Relative Year')
    plt.legend()
    plt.grid(True)
    plt.show()


    # plot predictions vs. u5 mortality
    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_df['Under-5 Mortality'], predict_df['BLFMM_pred'], label='Predicted BLFMM', color='red')
    plt.plot(lowess_results_u5[:, 0], lowess_results_u5[:, 1], 'g-', label='Lowess Fit', linewidth=2)
    plt.xlabel('Under-5 Mortality')
    plt.ylabel('Predicted BLFMM')
    plt.title('Predicted BLFMM by Under-5 Mortality')
    plt.legend()
    plt.grid(True)
    plt.show()

    # plot predictions vs. infant mortality
    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_df['Infant Mortality'], predict_df['BLFMM_pred'], label='Predicted BLFMM', color='red')
    plt.plot(lowess_results_infant[:, 0], lowess_results_infant[:, 1], 'g-', label='Lowess Fit', linewidth=2)
    plt.xlabel('Infant Mortality')
    plt.ylabel('Predicted BLFMM')
    plt.title('Predicted BLFMM by Infant Mortality')
    plt.legend()
    plt.grid(True)
    plt.show()

    # truncate life expectancy data
    predict_df['Life Expectancy'] = filtered_df['Life Expectancy']
    predict_df = predict_df[predict_df['Life Expectancy'] >= 30]
    lowess_results_le = lowess(predict_df['BLFMM_pred'], predict_df['Life Expectancy'], frac=0.3)

    # plot predictions vs life expectancy
    plt.figure(figsize=(10, 6))
    plt.scatter(predict_df['Life Expectancy'], predict_df['BLFMM_pred'], label='Predicted BLFMM', color='red')
    plt.plot(lowess_results_le[:, 0], lowess_results_le[:, 1], 'g-', label='Lowess Fit', linewidth=2)
    plt.xlabel('Life Expectancy')
    plt.ylabel('Predicted BLFMM')
    plt.title('Predicted BLFMM by Life Expectancy')
    plt.legend()
    plt.grid(True)
    plt.show()


def run_simple_regression(filtered_df):
    formula = 'BLFMM ~ Year + C(Country)'
    model = smf.ols(formula=formula, data=filtered_df).fit()

    print(model.summary())

    # Extract the residuals
    residuals = model.resid
    filtered_df['residuals'] = residuals

    # Plot residuals against year
    plt.scatter(filtered_df['relative_year'], filtered_df['residuals'], label='Residuals')
    plt.xlabel('Relative Year')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Relative Year')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.legend()
    plt.show()

    return model


def main():
    df = pd.read_excel("Input_DFs/BLFMM by Country 1950-2020 + modal age + mortality.xlsx")

    transition_years_df = has_single_transition(df)

    model = run_regression(transition_years_df)

    # plot_fitted_curve(transition_years_df, model)
    # run_simple_regression(transition_years_df)

if __name__ == '__main__':
    main()
