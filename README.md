# Health Inequality Index

## Excel Files
Many of the excel files in this repository are too large to store on git and appear corrupted, so use this link to access any files that are inaccessible here, under the same names:

https://drive.google.com/file/d/1cjOWXxqlP9LbWZ0voeG3qJiNZfPYZ_lU/view?usp=share_link

## Input Dataframes

Some input dataframes are too large to be stored in git and are available for download via Google Drive. To properly reproduce the projets, they should be added at the following paths:

- [`"UNPD Inequality Index All Countries/Input_Dataframes/Country_Data_1950-1985.xlsx"`](https://docs.google.com/spreadsheets/d/1ZwCYy6AKnGAjSgxOU7y5GtQa7wER5Qwt/edit?usp=sharing&ouid=104802165490528333494&rtpof=true&sd=true)
- [`"UNPD Inequality Index All Countries/Input_Dataframes/Country_Data_1986-2019.xlsx"`](https://docs.google.com/spreadsheets/d/1MRSf6KKv5kG5U77mZTABa5xhDSMLj35Q/edit?usp=sharing&ouid=104802165490528333494&rtpof=true&sd=true)


## Health Inequality Project Update 09/2024

We have created our BLFMM index and replicated methodology for Gini and Jamison indices — see code in “data-agg.” We’ve run longitudinal plots for each country and index from 1950-2022 to visualize changes over time in specific countries, displayed in “data-analysis/Index Plots Over Time”. We remade these plots only for countries that transition away from having a modal age at death of 0 during the time-frame, adjusting the axis so that a country’s “transition year” is Year 0. These are shown in “data-analysis/Adjusted Year Plots”. A figure showing all three indices for all years and all countries on a single plot can be found in “data-analysis/Results/all_indices_all_countries.png.” 

Kuznets updates: 
We noticed that in plotting BLFMM over time for transition countries, the graph increases steadily until the transition year, then jumps down significantly at the transition year, and then decreases steadily. Apart from the jump at the transition year, this looked like a kuznets pattern.
In an effort to make this pattern more recognizable by controlling for the transition jump, we trained a regression model with BLFMM as a function of: year FE, country FE, dummy variable for after mode transition, linear event/relative year, and the interaction between dummy for after * linear event year. We trained two models; one with just the equation above and another also including squared terms for the linear event year and interaction term. See “data-analysis/Control_Escape_Year.py”
We then used these model to predict BLFMM using actual values for linear event/relative year and the interaction term (and their squared terms). The BLFMM predictions were plotted and can be found in “data-analysis/Results/Regression.”

WDI updates:
Aggregated all WDI variables from World Bank in “data-analysis/Get Covariates.py.” Filtered indicators to include only those with <90% missingness. We then found the collinearity between each of the resulting indicators, and for pairs with >85% collinearity, filtered out the indicator with more missingness. Among pairs with the same missingness, we manually selected the more general of the two (i.e. “Life Expectancy” would be selected over “Life Expectancy, Male”).
We then ran random forests on our filtered WDI list for 2019 in “data-analysis/main.py” for each index (Gini, Jamison, BLFMM). We ran separate RFs for high-infant mortality countries and low-infant mortality countries. Results are displayed in “data-analysis/RFs/RFs 191 Covariates.”
