import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


def ATE_with_S_learner(data, regressor, treatment_col='TREATMENT', outcome_col='OUTCOME'):
    """
    Calculate the Average Treatment Effect, using S-Learner approach.
    :param data: Pandas.Dataframe
    :param regressor: Regression model that supports fit and predict functions
    :param treatment_col: name of treatment column
    :param outcome_col: name of outcome column
    :return: ATE score
    """
    # Fit a model y=~f(x, t), t is a feature in the models
    XT = data.drop(columns=[outcome_col], inplace=False)
    y = data[outcome_col]
    fitted_regressor = regressor.fit(XT, y)

    # Create a storage for the potential outcomes
    potential_outcomes_df = pd.DataFrame(0, index=range(len(data)), columns=[treatment_col, outcome_col, "y0_hat", "y1_hat"])
    potential_outcomes_df[treatment_col] = data[treatment_col]
    potential_outcomes_df[outcome_col] = data[outcome_col]

    # create data copies as if the treatment was 0 or 1
    data_t0 = data.copy()
    data_t0[treatment_col] = 0
    data_t1 = data.copy()
    data_t1[treatment_col] = 1

    # Fill the potential outcomes with the observed value for the true treatment
    # and the prediction of the outcome for the opposite treatment
    potential_outcomes_df["y0_hat"] = fitted_regressor.predict(data_t0.drop(columns=[outcome_col]))
    potential_outcomes_df["y1_hat"] = fitted_regressor.predict(data_t1.drop(columns=[outcome_col]))
    for index, row in potential_outcomes_df.iterrows():
        if row[treatment_col] == 0:
            row["y0_hat"] = row['OUTCOME']
        else:  # row[treatment_col] == 1
            row["y1_hat"] = row['OUTCOME']

    # Calculate estimator for ATE
    outcomes_diffs = potential_outcomes_df["y1_hat"].to_numpy() - potential_outcomes_df["y0_hat"].to_numpy()
    return np.mean(outcomes_diffs)


def ATE_with_T_learner(data, regressor0, regressor1, treatment_col='TREATMENT', outcome_col='OUTCOME'):
    """
    Calculate the Average Treatment Effect, using T-Learner approach.
    :param data: Pandas.Dataframe
    :param regressor0: Regression model that supports fit and predict functions
    :param regressor1: Regression model that supports fit and predict functions
    :param treatment_col: name of treatment column
    :param outcome_col: name of outcome column
    :return: ATE score
    """
    # Fit two models y1=~f(x where t=1), y0=~f(x where t=0), t is not a feature in the models
    # T=0
    data_t0 = data[data[treatment_col] == 0]
    X_t0 = data_t0.drop(columns=[outcome_col], inplace=False)
    y_t0 = data_t0[outcome_col]
    fitted_regressor0 = regressor0.fit(X_t0, y_t0)
    # T=1
    data_t1 = data[data[treatment_col] == 1]
    X_t1 = data_t1.drop(columns=[outcome_col], inplace=False)
    y_t1 = data_t1[outcome_col]
    fitted_regressor1 = regressor1.fit(X_t1, y_t1)

    # Create a storage for the potential outcomes
    potential_outcomes_df = pd.DataFrame(0, index=range(len(data)), columns=[treatment_col, outcome_col, "y0_hat", "y1_hat"])
    potential_outcomes_df[treatment_col] = data[treatment_col]
    potential_outcomes_df[outcome_col] = data[outcome_col]

    # Fill the potential outcomes with the observed value for the true treatment
    # and the prediction of the outcome for the opposite treatment
    potential_outcomes_df["y0_hat"] = fitted_regressor0.predict(data.drop(columns=[outcome_col]))
    potential_outcomes_df["y1_hat"] = fitted_regressor1.predict(data.drop(columns=[outcome_col]))
    for index, row in potential_outcomes_df.iterrows():
        if row[treatment_col] == 0:
            row["y0_hat"] = row['OUTCOME']
        else:  # row[treatment_col] == 1
            row["y1_hat"] = row['OUTCOME']

    # Calculate estimator for ATE
    outcomes_diffs = potential_outcomes_df["y1_hat"].to_numpy() - potential_outcomes_df["y0_hat"].to_numpy()
    return np.mean(outcomes_diffs)


def ATE_with_matching(data, n_neighbors, treatment_col='TREATMENT', outcome_col='OUTCOME'):
    """
    Calculate the Average Treatment Effect, using Matching approach.
    :param data: Pandas.DataFrame
    :param n_neighbors: number of nearest counterfactual neighbor of each sample in treatment group
    :param treatment_col: name of treatment column
    :param outcome_col: name of outcome column
    :return: ATE score
    """

    # Separate the treatment and control groups
    treatment_group = data[data[treatment_col] == 1]
    control_group = data[data[treatment_col] == 0]

    # Extract the features for matching
    features = data.drop(columns=[treatment_col, outcome_col])

    # Fit the nearest neighbors model on the control group
    nn_control = NearestNeighbors(n_neighbors=n_neighbors)
    nn_control.fit(control_group[features.columns])

    # Find the nearest neighbors in the control group for each treated unit
    distances_control, indices_control = nn_control.kneighbors(treatment_group[features.columns])

    # Calculate the average outcome for the matched control units
    matched_outcomes_control = control_group.iloc[indices_control.flatten()][outcome_col].values.reshape(-1, n_neighbors)
    avg_matched_outcomes_control = matched_outcomes_control.mean(axis=1)

    # Fit the nearest neighbors model on the treatment group
    nn_treatment = NearestNeighbors(n_neighbors=n_neighbors)
    nn_treatment.fit(treatment_group[features.columns])

    # Find the nearest neighbors in the treatment group for each control unit
    distances_treatment, indices_treatment = nn_treatment.kneighbors(control_group[features.columns])

    # Calculate the average outcome for the matched treatment units
    matched_outcomes_treatment = treatment_group.iloc[indices_treatment.flatten()][outcome_col].values.reshape(-1, n_neighbors)
    avg_matched_outcomes_treatment = matched_outcomes_treatment.mean(axis=1)

    # Calculate the ATE
    ATE_treatment = treatment_group[outcome_col].mean() - avg_matched_outcomes_control.mean()
    ATE_control = avg_matched_outcomes_treatment.mean() - control_group[outcome_col].mean()

    # Calculate the weighted average of the ATEs
    n_treatment = len(treatment_group)
    n_control = len(control_group)
    weighted_ATE = (ATE_treatment * n_treatment + ATE_control * n_control) / (n_treatment + n_control)

    return weighted_ATE
