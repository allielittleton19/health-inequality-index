import pandas as pd



# Function to find Var Selected for df_2 based on df_1
def populate_var_selected(df_1, df_2):
    # Create a set of tuples (Var A, Var B, Var Selected) and (Var B, Var A, Var Selected) from df_1 for easy lookup
    df_1_pairs = {}
    for _, row in df_1.iterrows():
        key1 = (row['Var A'], row['Var B'])
        key2 = (row['Var B'], row['Var A'])
        df_1_pairs[key1] = row['Var Selected']
        df_1_pairs[key2] = row['Var Selected']

    # Populate the 'Var Selected' column in df_2
    df_2['Var Selected'] = df_2.apply(lambda row: df_1_pairs.get((row['Var A'], row['Var B']), ''), axis=1)

    return df_2


def main():
    df_1 = pd.read_excel("90% Collinear Pairs with Same Missingness.xlsx")
    df_2 = pd.read_excel("85% Collinear Pairs with Same Missingness.xlsx")

    populated_df = populate_var_selected(df_1, df_2)

    populated_df.to_excel("85% Collinear Pairs with Selected Var.xlsx")


if __name__ == '__main__':
    main()
