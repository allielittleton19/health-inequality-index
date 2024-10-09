import matplotlib.pyplot as plt


import matplotlib.pyplot as mp
import pandas as pd
import seaborn as sb
import numpy as np
import statsmodels.api as sm
import matplotlib.cm as cm


def plot_spaghetti(df, title, name):
    color_palette = plt.cm.get_cmap('tab20', len(df.index))

    plt.figure(figsize=(10, 6))

    x_original = np.arange(df.columns.min(), df.columns.max() + 1)
    for idx, column in enumerate(df.index):
        y_original = df.loc[column]
        x_smooth = np.linspace(x_original.min(), x_original.max(), 500)
        y_smooth = np.interp(x_smooth, x_original, y_original)

        # Line width set to a medium thickness, between the previous two plots
        plt.plot(x_smooth, y_smooth, label=column, linewidth=1.75, color=color_palette(idx))

    plt.xlabel('Year')
    plt.ylabel('Rank Importance')
    plt.title(title)
    plt.gca().invert_yaxis()  # Reversing the y-axis
    plt.xticks(df.columns)  # Setting x-axis ticks to show each year
    plt.yticks(range(1, 12))  # Setting y-axis ticks for each rank (1 to 11)

    plt.legend(loc='center left', bbox_to_anchor=(1, .5))
    plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust the rect parameter to make space for the legend

    plt.savefig(name, format='png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    df_himr = pd.read_excel("HIMR_rankings_by_year.xlsx", index_col=0)
    df_limr = pd.read_excel("LIMR_rankings_by_year.xlsx", index_col=0)

    plot_spaghetti(df_himr, 'HIMR Variable Ranks Over Years', 'HIMR_Sphaghetti.png')
    plot_spaghetti(df_limr, 'LIMR Variable Ranks Over Years', 'LIMR_Sphaghetti.png')


    #correlogram(himr_data)
    #correlogram(limr_data)
    #regression(himr_data)
    #regression(limr_data)



    #lasso(himr_data)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
