import os
import ast
import glob
import pandas as pd
from scipy.stats import f_oneway, friedmanchisquare

def group_stats(sem_files):
    dataframes = []
    salience_by_participant = []
    for file in sem_files:
        current = pd.read_csv(file)
        dataframes.append(current)
    for df in dataframes:
        salience_by_participant.append(
            df['relative_salience'].apply(lambda x: ast.literal_eval(x)).tolist())
    # Perform the ANOVA test
    f_value, p_value = f_oneway(*salience_by_participant)
    print("ANOVA test results:")
    print("F-value:", f_value)
    print("p-value:", p_value)


def indiv_stats(sem_files):
    dataframes = []
    salience_by_session = []
    for file in sem_files:
        current = pd.read_csv(file)
        dataframes.append(current)

    # Extract the relative_salience variable for each session of each participant
    for participant in range(6):
        salience_by_session.append([])
        for df in dataframes:
            sessions = df[df['participant_number'] == participant]['relative_salience']
            salience_by_session[participant].append(sessions)

    # Perform the repeated measures ANOVA test
    f_value, p_value = friedmanchisquare(*salience_by_session)
    print("Repeated measures ANOVA test results:")
    print("F-value:", f_value)
    print("p-value:", p_value)

if __name__ == "__main__":
    base_path = os.getcwd()
    sem = os.path.join(base_path, '..', 'semPreProc')
    sem_files = sorted(glob.glob(os.path.join(sem, '*.csv')))
    group_stats(sem_files)