import pandas as pd
from folktables import ACSDataSource, generate_categories, BasicProblem


def load_acs_dataset(survey_year, states=None, horizon='1-Year', survey='person', download=False):
    """ load the data from the folktables API """
    # load the data
    data_source = ACSDataSource(survey_year=survey_year, horizon=horizon, survey=survey)
    data = data_source.get_data(states=states, download=download)
    definition_df = data_source.get_definitions(download=download)
    categories = generate_categories(definition_df=definition_df, features=list(data.columns))

    definitions_dict = _to_dictionary(definition_df)  # Construct variable explanations dictionary

    return data, definitions_dict, categories


def _to_dictionary(var_definitions):
    """ Construct variable explanations dictionary """
    var_exp_df = var_definitions.copy()
    var_exp_df.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G']  # dummy names
    var_exp_df = var_exp_df[var_exp_df['A'] == 'NAME'][['B', 'E']]
    var_exp_dict = {}
    for _, row in var_exp_df.iterrows():
        var_exp_dict[row['B']] = row['E']
    return var_exp_dict


def preprocess(data, categories):
    """ Preprocess the data """
    # define the treatment and outcome. MIL == military service. COW == class of worker
    data['TREATMENT'] = data.apply(lambda row: 1.0 if row['MIL'] in [2.0, 3.0] else 0.0, axis=1)
    data['OUTCOME'] = data.apply(lambda row: 1.0 if row['COW'] in [3.0, 4.0, 5.0] else 0.0, axis=1)

    y_name = 'OUTCOME'
    features = list(set(data.columns) - {y_name})
    basic_problem = BasicProblem(features=features, target=y_name)
    X_df, y_df, _ = basic_problem.df_to_pandas(df=data, categories=categories, dummies=True)
    Xy_df = pd.concat([X_df, y_df], axis=1)
    Xy_df = Xy_df.apply(lambda col: col.astype(float))

    return Xy_df
