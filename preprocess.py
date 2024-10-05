import pandas as pd
from folktables import ACSDataSource, generate_categories, BasicProblem


# todo document this file

def load_acs_dataset(survey_year, states=None, horizon='1-Year', survey='person', download=False, density=1, random_seed=1):
    """ load the data from the folktables API """

    def _to_dictionary(var_definitions):
        """ Construct variable explanations dictionary """
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
    """ Preprocess the data """

    def _rename_duplicate_columns(df):
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
        df.columns = cols
        return df

    # define the treatment and outcome. MIL == military service. COW == class of worker
    data['TREATMENT'] = data.apply(treatment_definition_func, axis=1)
    data['OUTCOME'] = data.apply(outcome_definition_func, axis=1)
    features_to_select.extend(['TREATMENT', 'OUTCOME'])
    data_filtered_cols = data[features_to_select]

    features = list(set(data_filtered_cols.columns) - {'OUTCOME'})
    basic_problem = BasicProblem(features=features, target='OUTCOME')
    X_df, y_df, _ = basic_problem.df_to_pandas(df=data_filtered_cols, categories=categories, dummies=True)
    Xy_df = pd.concat([X_df, y_df], axis=1)
    Xy_df = Xy_df.apply(lambda col: col.astype(float))

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
