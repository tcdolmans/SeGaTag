"""
 * @author [Tenzing Dolmans]
 * @email [t.c.dolmans@gmail.com]
 * @create date 2023-06-30 17:23:28
 * @modify date 2023-06-30 17:23:28
 * @desc [description]
"""
import os
import ast
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_sem_rs(sem_files, mode="session"):
    """
    Extracts the semantic vectors and relative salience from the semantic
    analysis files. The semantic vectors and relative salience are averaged
    per participant and returned in a dictionary.
    "Mode" determines whether the participant number is averaged as a whole
    or per session.
    Input:
        sem_files: list of semantic analysis files
        mode: "session" or "participant"
    Output:
        participant_dict: dictionary with participant numbers as keys and
        semantic vectors and relative salience as values
    """
    participant_dict = {}
    sem_vectors = []
    relative_salience = []
    p_num = ''

    for file in sem_files:
        current = pd.read_csv(file)
        if mode == "session":
            p_num = str(current['participant_number'][0][:7])
        elif mode == "participant":
            p_num = str(current['participant_number'][0][:4])
        else:
            raise ValueError("Mode must be either 'session' or 'participant'")

        sem_vectors = current['sem_vector'].apply(ast.literal_eval)
        sem_vectors = np.mean(sem_vectors.tolist(), axis=0)
        relative_salience = current['relative_salience'].apply(ast.literal_eval)
        relative_salience = np.mean(relative_salience.tolist(), axis=0)

        if p_num in participant_dict:
            participant_dict[p_num]['sem'].append(sem_vectors)
            participant_dict[p_num]['rs'].append(relative_salience)
        else:
            participant_dict[p_num] = {'sem': [sem_vectors], 'rs': [relative_salience]}

    for key in participant_dict.keys():
        sem = participant_dict[key]['sem']
        sem = np.mean(sem, axis=0)
        participant_dict[key]['sem'] = sem

        rs = participant_dict[key]['rs']
        rs = np.mean(rs, axis=0)
        participant_dict[key]['rs'] = rs
    return participant_dict


def plot_spider_chart(participant_dict):
    """
    Plots a spider chart for each participant in the participant dictionary.
    Depends on the mode of extract_sem_rs, the participant number is either
    averaged per session or as a whole.
    Input:
        participant_dict: dictionary with participant numbers as keys and
        semantic vectors and relative salience as values
    Output:
        None, but it makes pretty pictures!
    """
    rs = []
    for key in participant_dict.keys():
        sem = participant_dict[key]['sem']
        rs = participant_dict[key]['rs']

        # Number of variables
        num_vars = len(rs)

        # Calculate angles for each variable
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        # angles += angles[:1]  # Include the starting angle to close the chart

        # Create the spider chart plot
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'polar': True})
        ax.fill(angles, rs, color='skyblue', alpha=0.5)  # Fill the area inside the chart

        # Set the labels for each variable
        ax.set_xticks(angles)
        ax.set_xticklabels(range(num_vars))

        # Set the title
        title = 'Participant ' + str(key)
        ax.set_title(title, size=20)

        # Set the gridlines
        ax.grid(True)

        # Display the spider chart
        plt.show()


if __name__ == "__main__":
    base_path = os.getcwd()
    sem = os.path.join(base_path, 'semPreProc')
    sem_files = sorted(glob.glob(os.path.join(sem, '*.csv')))
    participant_dict = extract_sem_rs(sem_files, mode="participant")
    plot_spider_chart(participant_dict)
