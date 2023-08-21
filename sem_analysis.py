"""
 * @author [Tenzing Dolmans]
 * @email [t.c.dolmans@gmail.com]
 * @create date 2023-06-30 17:23:28
 * @modify date 21-08-2023 15:31:35
 * @desc [description]
"""
import ast
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_sem_rs(sem_files, mode="session"):
    """
    Extracts the semantic vectors and relative salience from the semantic
    analysis files. The semantic vectors and relative salience are averaged
    per participant and returned in a dictionary.
    "Mode" is either "session" or "participant" and determines whether
    the participant number is averaged as a whole or per session.

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
    p_num = ""

    for file in sem_files:
        current = pd.read_csv(file)
        if mode == "session":
            p_num = str(current["participant_number"][0][:7])
        elif mode == "participant":
            p_num = str(current["participant_number"][0][:4])
        else:
            raise ValueError("Mode must be either 'session' or 'participant'")

        sem_vectors = current["sem_vector"].apply(ast.literal_eval)
        sem_vectors = np.mean(sem_vectors.tolist(), axis=0)
        relative_salience = current["relative_salience"].apply(ast.literal_eval)
        relative_salience = np.mean(relative_salience.tolist(), axis=0)

        if p_num in participant_dict:
            participant_dict[p_num]["sem"].append(sem_vectors)
            participant_dict[p_num]["rs"].append(relative_salience)
        else:
            participant_dict[p_num] = {"sem": [sem_vectors], "rs": [relative_salience]}

    for key in participant_dict.keys():
        sem = participant_dict[key]["sem"]
        sem = np.mean(sem, axis=0)
        participant_dict[key]["sem"] = sem

        rs = participant_dict[key]["rs"]
        rs = np.mean(rs, axis=0)
        participant_dict[key]["rs"] = rs
    return participant_dict


def convert_p_num_to_int(p_num):
    """
    Converts the participant number to an integer value. This is needed for later
    code that relies on values being type(int).

    Input:
        p_num: participant number as a string
    Output:
        p_num: participant number as an integer
        session: session number as an integer
    """
    p_num = p_num[1:4] + p_num[5:]
    session = p_num[3:]
    p_num = p_num[:-2]
    if session.endswith("a"):
        session = session[:-1] + "1"
    elif session.endswith("b"):
        session = session[:-1] + "2"
    elif session.endswith("c"):
        session = session[:-1] + "3"
    else:
        raise ValueError("Participant number must end with 'a', 'b' or 'c'")
    return int(p_num), int(session)


def generate_p_samples(sem_file):
    """
    Takes an input CSV file and generates a list that contains the lines
    of the CSV as samples. Each sample in the list contains following information:
     - Participant number: str
     - Image number: str
     - Semantic vector (contained in the 'sem_vector' column): list, size 12
     - Presence of dimension (contained in the 'dim_presence' column): list, size 12
     - Relative salience (contained in the 'relative_salience' column): list, size 12
    Input:
        sem_file: CSV file of one participant
    Output:
        samples: list containing the samples of the CSV file, as described above
    """
    df = pd.read_csv(sem_file)
    samples = []
    for i, row in df.iterrows():
        # Extract the required columns from the DataFrame
        p_num, session = convert_p_num_to_int(row["participant_number"])

        # Use eval() to convert the string to a list, since the storage is a list of ints
        # and we want to get those ints back.
        sample = {
            "p_num": p_num,
            "session": session,
            "img": int(row["img"][:4]),
            "sem_vector": eval(row["sem_vector"]),
            "dim_pres": eval(row["dim_presence"]),
            "rel_sal": eval(row["relative_salience"]),
        }
        samples.append(sample)
    return samples


def generate_master_sample(sem_files):
    """
    Generates a master sample list that contains all samples from all CSV files.
    All items in the list are dictionaries, so they can easily be sliced. See the
    generate_p_samples() function for more information on the dictionary keys.

    Input:
        sem_files: list of CSV files
    Output:
        master_samples: list containing all samples from all CSV files
    """
    master_samples = []
    # Iterate over files in the folder
    for file in sem_files:
        samples = generate_p_samples(file)
        master_samples.extend(samples)
    return master_samples


def get_unique_values(master_samples, verbose=False):
    """
    Prints the unique values for participant number, session and image number.
    Mainly used to query the master sample list for slices.

    Input:
        master_samples: list containing all samples from all CSV files
        verbose: if True, prints the unique values
    Output:
        unique_p_nums: set of unique participant numbers
        unique_sessions: set of unique session numbers
        unique_imgs: set of unique image numbers
    """
    unique_p_nums = set()
    unique_sessions = set()
    unique_imgs = set()

    for sample in master_samples:
        unique_p_nums.add(sample["p_num"])
        unique_sessions.add(sample["session"])
        unique_imgs.add(sample["img"])
    if verbose:
        print("Unique Participant Numbers:")
        for p_num in unique_p_nums:
            print(p_num)

        print("\nUnique sessions Numbers:")
        for sesh in unique_sessions:
            print(sesh)

        print("\nUnique Image Numbers:")
        for img in unique_imgs:
            print(img)
    return unique_p_nums, unique_sessions, unique_imgs


def plot_spider_chart(participant_dict, to_plot="rs"):
    """
    Plots a spider chart for each participant in the participant dictionary.
    Depends on the mode of 'extract_sem_rs', the participant's data is either
    averaged per session or as a whole.
    Input:
        participant_dict: dictionary with participant numbers as keys and
        semantic vectors and relative salience as values
        to_plot: "sem" or "rs", determines which variable to plot
    Output:
        None, but it makes pretty pictures!
    """
    if not to_plot == "rs" or to_plot == "sem":
        raise ValueError("to_plot must be either 'rs' or 'sem'")
    for key in participant_dict.keys():
        plot_data = participant_dict[key][to_plot]

        # Number of variables
        num_vars = len(plot_data)

        # Calculate angles for each variable
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        # angles += angles[:1]  # Include the starting angle to close the chart

        # Create the spider chart plot
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
        ax.fill(
            angles, plot_data, color="skyblue", alpha=0.5
        )  # Fill the area inside the chart

        # Set the labels for each variable
        ax.set_xticks(angles)
        ax.set_xticklabels(range(num_vars))

        # Set the title
        title = "Participant " + str(key)
        ax.set_title(title, size=20)

        # Set the gridlines
        ax.grid(True)

        # Display the spider chart
        plt.show()


if __name__ == "__main__":
    # File paths, please read the readme for more information on the folder structure
    base_path = os.getcwd()
    sem = os.path.join(base_path, "semPreProc")
    sem_files = sorted(glob.glob(os.path.join(sem, "*.csv")))

    master = generate_master_sample(sem_files)
    # print some info about the master sample
    print("Master sample length:", len(master))
    print("Master sample keys:", master[0].keys())

    # Extract unique values. This is useful for slicing the master sample.
    # Please read get_unique_values() for more information. Also expand it!
    unique_p_nums, unique_sessions, unique_imgs = get_unique_values(
        master, verbose=False
    )
    print("Number of unique participants:", len(unique_p_nums))
    print("Number of unique sessions:", len(unique_sessions))
    print("Number of unique images:", len(unique_imgs))

    # You can subset the master sample by any of the categories as described in the readme.
    # For example, to get all samples where the face dimension is present:
    # face:        sample["dim_pres"][0] > 0
    # # or motion: sample["dim_pres"][4] > 0, etc.
    face_subset = [sample for sample in master if sample["dim_pres"][0] > 0]
    motion_subset = [sample for sample in master if sample["dim_pres"][4] > 0]

    # Then, rerun the get_unique_values() function to get the unique values for the subset
    unique_p_nums, unique_sessions, unique_imgs = get_unique_values(
        motion_subset, verbose=False
    )
    # And print information about the subset
    print("Number of unique participants:", len(unique_p_nums))
    print("Number of unique sessions:", len(unique_sessions))
    print("Number of unique images:", len(unique_imgs))

    participant_dict = extract_sem_rs(sem_files, mode="participant")
    print("Participant dictionary keys:")
    for key in participant_dict.keys():
        print(key)
    plot_spider_chart(participant_dict, to_plot="rs")
