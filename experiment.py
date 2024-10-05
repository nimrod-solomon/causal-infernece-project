from preprocess import *

SELECTED_FEATURES = pd.read_csv("./supplementary materials/selected variables.csv")["code"].to_list()
N_BOOTSTRAP_TRIALS = 100
SAMPLES_PER_TRIAL = 10000

if __name__ == '__main__':
    raw_data, variables_definitions, answers_parsing = load_acs_dataset(survey_year='2022', states=['AK'])  # todo remove states=AK
    processed_df = preprocess(raw_data, answers_parsing)


