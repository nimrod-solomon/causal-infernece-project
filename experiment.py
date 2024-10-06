import numpy as np
import time
from preprocess import load_acs_dataset, preprocess
from potential_outcomes import ATE_with_matching, ATE_with_S_learner, ATE_with_T_learner
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Constants
SELECTED_FEATURES = pd.read_csv("./supplementary materials/selected variables.csv")["code"].to_list()
N_BOOTSTRAP_TRIALS = 10
DENSITY_PER_TRIAL = 0.0001
TREATMENT_DEFINITION = lambda unit: 1.0 if unit['MIL'] in [2.0, 3.0] else 0.0
OUTCOME_DEFINITION = lambda unit: 1.0 if unit['COW'] in [3.0, 4.0, 5.0] else 0.0


# Run Experiment
if __name__ == '__main__':
    ate_s_learner_list = []
    ate_t_learner_list = []
    ate_matching_list = []

    i = 0  # iteration number (updates after each successful iteration)
    sample_seed = 0  # another seed for the sample (updates after each iteration)
    while i < N_BOOTSTRAP_TRIALS:
        start_time = time.time()

        # Sample small portion of the data
        raw_data, variables_definitions, answers_parsing = load_acs_dataset(survey_year='2022', density=DENSITY_PER_TRIAL, random_seed=sample_seed)
        processed_df = preprocess(data=raw_data, categories=answers_parsing, treatment_definition_func=TREATMENT_DEFINITION,
                                  outcome_definition_func=OUTCOME_DEFINITION, features_to_select=SELECTED_FEATURES)

        # Create regressors for S\T Learners
        regressor = RandomForestRegressor(random_state=i)
        regressor0 = LogisticRegression(random_state=i)
        regressor1 = LogisticRegression(random_state=i)

        try:
            # Calculate ATE in several ways
            ate_s_learner = ATE_with_S_learner(data=processed_df, regressor=regressor)
            ate_t_learner = ATE_with_T_learner(data=processed_df, regressor0=regressor0, regressor1=regressor1)
            ate_matching = ATE_with_matching(data=processed_df, n_neighbors=1)

        except ValueError:  # it means the sample doesnt contain two classes, only one
            continue  # skip this sample

        else:  # print ans save the ATEs
            print(f"Trial No. {i+1}:", end="\t")
            print(f"ATE with S-Learner: {ate_s_learner}", end="\t")
            print(f"ATE with T-Learner: {ate_t_learner}", end="\t")
            print(f"ATE with Matching: {ate_matching}", end="\t")
            ate_s_learner_list.append(ate_s_learner)
            ate_t_learner_list.append(ate_t_learner)
            ate_matching_list.append(ate_matching)

            end_time = time.time()
            iteration_time = end_time - start_time
            print(f"Iteration Time: {iteration_time:.2f} seconds")
            i += 1

        finally:
            sample_seed += 1

    # Print summary of experiment
    print()
    print("Summary:")
    print("S-Learner Method:")
    ate_s_learner_nparray = np.array(ate_s_learner_list)
    mean_value = np.mean(ate_s_learner_nparray)
    quantile_05 = np.quantile(ate_s_learner_nparray, 0.05)
    quantile_95 = np.quantile(ate_s_learner_nparray, 0.95)
    print(f"Avg ATE over {N_BOOTSTRAP_TRIALS} trials: {mean_value}. Confidence Interval: [{quantile_05},{quantile_95}]")

    print("T-Learner Method:")
    ate_t_learner_nparray = np.array(ate_t_learner_list)
    mean_value = np.mean(ate_t_learner_nparray)
    quantile_05 = np.quantile(ate_t_learner_nparray, 0.05)
    quantile_95 = np.quantile(ate_t_learner_nparray, 0.95)
    print(f"Avg ATE over {N_BOOTSTRAP_TRIALS} trials: {mean_value}. Confidence Interval: [{quantile_05},{quantile_95}]")

    print("Matching Method:")
    ate_matching_nparray = np.array(ate_matching_list)
    mean_value = np.mean(ate_matching_nparray)
    quantile_05 = np.quantile(ate_matching_nparray, 0.05)
    quantile_95 = np.quantile(ate_matching_nparray, 0.95)
    print(f"Avg ATE over {N_BOOTSTRAP_TRIALS} trials: {mean_value}. Confidence Interval: [{quantile_05}, {quantile_95}]")
