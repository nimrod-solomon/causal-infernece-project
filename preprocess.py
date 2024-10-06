import pandas as pd
from folktables import ACSDataSource, generate_categories, BasicProblem


def load_acs_dataset(survey_year, states=None, horizon='1-Year', survey='person', download=False, density=1, random_seed=1):
    """
    Loads data from the American Community Survey (ACS) using the Folktables API.

    :param survey_year: The year of the ACS survey to load data from (int).
    :param states: A list of states to include in the data. If None, data from all states is loaded (list, optional).
    :param horizon: The time horizon of the ACS data (e.g., '1-Year', '3-Year') (str, optional).
    :param survey: The type of ACS survey (e.g., 'person', 'household') (str, optional).
    :param download: A boolean indicating whether to download the data locally (bool, optional).
    :param density: The density of the ACS data (e.g., 1, 2, 3) (int, optional).
    :param random_seed: A random seed used for sampling data if `density` is less than 1 (int, optional).

    :returns: tuple: A tuple containing:
            - data (pd.DataFrame): A Pandas DataFrame containing the loaded ACS data.
            - definitions_dict (dict): A dictionary mapping variable names to their definitions.
            - categories (list): A list of categories associated with the variables in the data.
    """

    def _to_dictionary(var_definitions):
        """
        Constructs a dictionary mapping variable names to their definitions.

        :param: var_definitions: A DataFrame containing the variable definitions (pd.DataFrame).

        :returns: A dictionary mapping variable names to their definitions (dict).
        """
        var_exp_df = var_definitions.copy()
        var_exp_df.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G']  # dummy names
        var_exp_df = var_exp_df[var_exp_df['A'] == 'NAME'][['B', 'E']]
        var_exp_dict = {}
        for _, row in var_exp_df.iterrows():
            var_exp_dict[row['B']] = row['E']
        return var_exp_dict

    # load the data
    data_source = ACSDataSource(survey_year=survey_year, horizon=horizon, survey=survey)
    data = data_source.get_data(states=states, download=download, density=density, join_household=True, random_seed=random_seed)
    definition_df = data_source.get_definitions(download=download)
    categories = generate_categories(definition_df=definition_df, features=list(data.columns))

    definitions_dict = _to_dictionary(definition_df)  # Construct variable explanations dictionary

    return data, definitions_dict, categories


def preprocess(data, categories, treatment_definition_func, outcome_definition_func, features_to_select):
    """
    Preprocesses the data for machine learning analysis.

    :param data: A pandas DataFrame containing the raw data.
    :param categories: A list of column names representing categorical features.
    :param treatment_definition_func: A function that defines the treatment variable based on the data.
    :param outcome_definition_func: A function that defines the outcome variable based on the data.
    :param features_to_select: A list of column names specifying the features to be included in the analysis.

    :returns: A preprocessed pandas DataFrame ready for machine learning.
    """

    def _rename_duplicate_columns(df):
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
        df.columns = cols
        return df

    # Define the treatment and outcome variables
    data['TREATMENT'] = data.apply(treatment_definition_func, axis=1)
    data['OUTCOME'] = data.apply(outcome_definition_func, axis=1)

    # Select the specified features
    features_to_select.extend(['TREATMENT', 'OUTCOME'])
    data_filtered_cols = data[features_to_select]
    features = list(set(data_filtered_cols.columns) - {'OUTCOME'})

    # Create a BasicProblem instance
    basic_problem = BasicProblem(features=features, target='OUTCOME')

    # Convert the data to pandas DataFrames, data types to float
    X_df, y_df, _ = basic_problem.df_to_pandas(df=data_filtered_cols, categories=categories, dummies=True)
    Xy_df = pd.concat([X_df, y_df], axis=1)
    Xy_df = Xy_df.apply(lambda col: col.astype(float))

    # Rename and delete duplicate columns, handle missing values
    Xy_df = _rename_duplicate_columns(Xy_df)

    for col in Xy_df.columns:
        if "TREATMENT_" in str(col) or "OUTCOME_" in str(col):
            Xy_df.drop(columns=[col], inplace=True)
            continue
        if Xy_df[col].dtype == object:  # Categorical column
            Xy_df[col] = Xy_df[col].fillna(0)
        else:  # Numerical column
            Xy_df[col] = Xy_df[col].fillna(Xy_df[col].mean())

    return Xy_df
