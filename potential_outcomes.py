import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
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
    # Fit a model y=~f(x,t) with t as a feature on the entire sample
    y = data[outcome_col]
    X = data.drop(columns=[outcome_col], inplace=False)
    fitted_regressor = regressor.fit(X, y)

    # predict f(x,1) - f(x,0)
    X_t1 = X.copy()
    X_t1['TREATMENT'] = 1
    predictions_t1 = fitted_regressor.predict(X_t1)
    X_t0 = X.copy()
    X_t0['TREATMENT'] = 0
    predictions_t0 = fitted_regressor.predict(X_t0)

    # calculate estimator for ATE
    predictions_t1_minus_t0 = predictions_t1 - predictions_t0
    return np.mean(predictions_t1_minus_t0)


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
    # Fit a model on each treatment group
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

    # create a storage for the potential outcomes
    potential_outcomes_df = pd.DataFrame(0, index=range(len(data)), columns=[treatment_col, outcome_col, "y0_hat", "y1_hat"])
    potential_outcomes_df[treatment_col] = data[treatment_col]
    potential_outcomes_df[outcome_col] = data[outcome_col]
    potential_outcomes_df["y0_hat"] = fitted_regressor0.predict(data.drop(columns=[outcome_col]))
    potential_outcomes_df["y1_hat"] = fitted_regressor1.predict(data.drop(columns=[outcome_col]))

    for index, row in potential_outcomes_df.iterrows():
        if row[treatment_col] == 0:
            row["y0_hat"] = row['OUTCOME']
        else:  # row[treatment_col] == 1
            row["y1_hat"] = row['OUTCOME']

    # average the diff
    treatment_diffs = potential_outcomes_df["y1_hat"].to_numpy() - potential_outcomes_df["y0_hat"].to_numpy()
    return np.mean(treatment_diffs)


def _ATE_with_matching(data, n_neighbors, treatment_col='TREATMENT', outcome_col='OUTCOME'):
    """
    Calculate the Average Treatment Effect, using Matching approach.
    :param data: Pandas.Dataframe
    :param n_neighbors: number of nearest counterfactual neighbor of each sample in treatment group
    :param treatment_col: name of treatment column
    :param outcome_col: name of outcome column
    :return: ATE score
    """
    def _CATE_by_treatment_group(treatment_group):
        # separate data to X,T,Y
        Y_col = data[outcome_col]
        T_col = data[treatment_col]

        # normalize data (for distance calculations purposes)
        XT = data.drop(columns=[outcome_col], inplace=False)
        XT_cols = XT.columns
        XT_scaled = minmax_scale(XT)
        XT_scaled = pd.DataFrame(XT_scaled, columns=XT_cols)

        # split data to X when T=0 and X when T=1
        XT_scaled_t0 = XT_scaled[XT_scaled[treatment_col] == 0.0]
        XT_scaled_t1 = XT_scaled[XT_scaled[treatment_col] == 1.0]
        X_scaled_t0 = XT_scaled_t0.drop(columns=[treatment_col])
        X_scaled_t1 = XT_scaled_t1.drop(columns=[treatment_col])

        # find indices of closest neighbors for each x for which T=1, in the group of x's where T=0
        nn_model = NearestNeighbors(n_neighbors=n_neighbors)
        if treatment_group == 1:
            nn_model.fit(X_scaled_t0)
            _, indices = nn_model.kneighbors(X_scaled_t1)
        else:  # treatment_group == 0
            nn_model.fit(X_scaled_t1)
            _, indices = nn_model.kneighbors(X_scaled_t0)

        # extract the Y values of these indices
        y_col_list = list(Y_col)
        t_col_list = list(T_col)
        Ys_of_t0 = [y_col_list[i] for i in range(len(y_col_list)) if t_col_list[i] == 0]
        Ys_of_t1 = [y_col_list[i] for i in range(len(y_col_list)) if t_col_list[i] == 1]

        if treatment_group == 1:
            Ys_of_t1 = np.array(Ys_of_t1)
            Ys_of_neighbors = Ys_of_t0
        else:
            Ys_of_t0 = np.array(Ys_of_t0)
            Ys_of_neighbors = Ys_of_t1

        yjs = np.zeros(indices.shape)
        for i in range(len(yjs)):  # for each sample
            for j in range(len(yjs[0])):  # for each of the sample's nearest neighbors
                yjs[i][j] = Ys_of_neighbors[indices[i][j]]  # get its y value

        # average the Y values
        avg_yjs = np.average(yjs, axis=1)

        # return the avg diff between the true Y value and the avg value of counterfactual neighbors
        if treatment_group == 1:
            return np.mean(Ys_of_t1 - avg_yjs)
        else:
            return np.mean(Ys_of_t0 - avg_yjs)

    cate0 = _CATE_by_treatment_group(treatment_group=0)
    prop_t0 = len(data[data[treatment_col] == 0]) / len(data)
    cate1 = _CATE_by_treatment_group(treatment_group=1)
    prop_t1 = len(data[data[treatment_col] == 1]) / len(data)

    return cate0 * prop_t0 + cate1 * prop_t1


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
