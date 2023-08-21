"""
 * @author [Tenzing Dolmans]
 * @email [t.c.dolmans@gmail.com]
 * @create date 2023-05-11 12:06:29
 * @modify date 21-08-2023 11:02:50
 * @desc [description]
"""
import glob
import os

import numpy as np
import pandas as pd
from utils import load_sem_data


def get_gaze_rows(gaze, msg, img):
    """
    Finds the gaze rows that correspond to the start and end times of the image
    and returns them along with the start and end times. Relies on Jukka's H5 files.

    Input:
        gaze: gaze data from the .h5 file
        msg: msg data from the .h5 file
        img: image name
    Output:
        gaze_rows: DataFrame with gaze data for a given image
        start_time: Start time of the image
        end_time: End time of the image
    """

    # TODO: Find the calibration data and use it to determine the parameters for the padding.
    # calibration = msg.loc[msg['msg'].str.contains('calibration')]

    # Start time when "onset_img_name" and end time when "offset_img_name" in 'msg' matches
    start_time_idx = msg.loc[
        msg["msg"].str.contains(f"onset.*{img}")
    ].first_valid_index()
    end_time_idx = msg.loc[
        msg["msg"].str.contains(f"offset.*{img}")
    ].first_valid_index()

    # Check whether the start and end times are empty, not every image is used as a stimulus
    if start_time_idx is None or end_time_idx is None:
        return None

    start_time = msg.loc[start_time_idx, "system_time_stamp"]
    end_time = msg.loc[end_time_idx, "system_time_stamp"]

    # In 'gaze', find the corresponding rows for the start and end times
    # gaze_rows contains all the data in the df for a given image
    gaze_rows = gaze.loc[
        (gaze["system_time_stamp"] >= start_time)
        & (gaze["system_time_stamp"] <= end_time)
    ]
    return gaze_rows, start_time, end_time


def max_consecutive_invalid(gaze_rows):
    """
    Calculates the number of consecutive and total invalid gaze points.

    Input:
        gaze_rows: DataFrame with gaze data for a given image
    Output:
        max_consecutive_count: Maximum number of consecutive invalid gaze points
        total_invalid: Total number of invalid gaze points
    """
    column = gaze_rows["left_gaze_point_valid"]
    total_invalid = column.value_counts().get(False, 0)
    consecutive_count = 0
    max_consecutive_count = 0
    for value in column:
        if value is False:
            consecutive_count += 1
        else:
            max_consecutive_count = max(max_consecutive_count, consecutive_count)
            consecutive_count = 0
    return max(max_consecutive_count, consecutive_count), total_invalid


def add_gaze_padding(x, y, padding=0):
    """
    Adds a padding to the gaze data, to account for calibration error.
    This is done by converting every value in x and y from single pixels to
    a circle of pixels around the original point, with radius equal to padding.
    Padding simply makes the list of coordinates longer, so nothing breaks later on.

    See the TODO in get_gaze_rows for details about determining padding from
    the calibration section of the experiment.

    Input:
        x: x-coordinates of gaze data
        y: y-coordinates of gaze data
        padding: padding in pixels
    Output:
        x_padded: x-coordinates of gaze data with padding
        y_padded: y-coordinates of gaze data with padding
        weights: weights of the points, inversely proportional to the distance to the center
                 weights can be used to scale the sem_vector, currently unimplemented
    """
    x_padded = []
    y_padded = []
    weights = []
    for xi, yi in zip(x, y):
        for dx in range(-padding, padding + 1):
            for dy in range(-padding, padding + 1):
                if (
                    dx * dx + dy * dy <= padding * padding
                ):  # This condition ensures we add points in a circle, not a square
                    x_padded.append(xi + dx)
                    y_padded.append(yi + dy)
                    weights.append(
                        1 / (1 + np.sqrt(dx * dx + dy * dy))
                    )  # This can be adjusted depending on your needs
    return np.array(x_padded), np.array(y_padded), np.array(weights)


def get_sem_vector(gaze_rows, label, padding=0):
    """
    For a given image and semantic label, calculate the overlap of gaze on semantic label.

    Inputs:
        gaze_rows: DataFrame with gaze data for a given image
        label: Semantic label for the image
        padding: Padding for gaze, in pixels
    Output:
        sem_vector: A vector of length 12, containing the fraction of gaze for each sem dimension
        dim_presence : A vector of length 12, containing total pixels in each sem dimension
    """
    # Select relevant columns from gaze data
    gaze_data = gaze_rows[
        [
            "left_gaze_point_on_display_area_x",
            "left_gaze_point_on_display_area_y",
            "right_gaze_point_on_display_area_x",
            "right_gaze_point_on_display_area_y",
        ]
    ]

    # Calculate the average gaze point of the two eyes, scale to image dimensions
    x = (
        (
            gaze_data["left_gaze_point_on_display_area_x"]
            + gaze_data["right_gaze_point_on_display_area_x"]
        )
        / 2
    ) * 1067
    y = (
        (
            gaze_data["left_gaze_point_on_display_area_y"]
            + gaze_data["right_gaze_point_on_display_area_y"]
        )
        / 2
    ) * 600
    x, y, weights = add_gaze_padding(x, y, padding=padding)

    # Lines below fill in 0,0 for NaN values in x and y.
    # TODO: consider the effect of this on the semantic vector in edge cases,
    #       since 0,0 is the top left corner of the image.
    x = np.nan_to_num(x).astype(int).tolist()
    y = np.nan_to_num(y).astype(int).tolist()

    # Make sure the values don't exceed the image dimensions (800, 600, so subtract 1)
    x = [min(val, 799) for val in x]
    y = [min(val, 599) for val in x]
    points = np.column_stack((x, y))

    sem_vector = []
    dim_presence = []

    for dim in label:
        matches = np.sum(dim[points[:, 0], points[:, 1]] != 0)
        sem_vector.append(matches / len(weights))
        dim_presence.append(sum(dim.flatten()))
    return sem_vector, dim_presence


def get_relative_salience(sem_vector, dim_presence):
    """
    Calculates the relative saliency for each semantic dimension.
    This is based on the following logic:
    Few pixel in a category but high gaze = high relative saliency
    Many pixels in a category but low gaze = low relative saliency
    etc.

    Inputs:
        sem_vector: A vector containing the fraction of gaze for each sem dimension
        dim_presence : A vector containing total pixels in each sem dimension
    Output:
        relative_salience: A vector containing the relative saliency for each sem dimension
    """
    max_dim_presence = 800 * 600  # Maximum possible value in dim_presence
    scaled_dim_presence = np.array(dim_presence) / max_dim_presence

    # If the maximum value in sem_vector is 0, the relative salience is 0 for all dimensions,
    # This results in a printed flag in the 'extract_et' function, which then discards the sample.
    if max(sem_vector) == 0:
        return np.zeros(len(sem_vector))

    relative_salience = np.divide(
        sem_vector,
        scaled_dim_presence,
        out=np.zeros_like(sem_vector),
        where=scaled_dim_presence != 0,
    )
    return list(relative_salience)


def extract_et(et_file, img_names, labels, padding, save_raw_gaze=False):
    """
    Extracts the eye-tracking data for a given participant and saves a CSV summary.
    If the save_raw_gaze flag is set to True, the raw gaze data for each image
    is saved in a separate csv file.

    Inputs:
        et_file: Path to the eye-tracking file
        img_names: List of image names
        labels: List of semantic labels
        padding: Padding for gaze, in pixels
        save_raw_gaze: Flag to save raw gaze data
    Saves:
        part_summary: DataFrame containing the summary data for the participant as CSV
        raw_gaze: only if save_raw_gaze is True, raw gaze data for each valid image as CSV
    """
    part_summary = pd.DataFrame(
        columns=[
            "participant_number",
            "img",
            "start_time",
            "end_time",
            "consecutive_invalid",
            "total_invalid",
            "sem_vector",
            "dim_presence",
            "relative_salience",
        ]
    )
    gaze = pd.read_hdf(et_file, "gaze")
    msg = pd.read_hdf(et_file, "msg")
    participant_number = os.path.splitext(os.path.basename(et_file))[0]
    print("Processing file: ", participant_number)

    for img in img_names:
        # Select correct label for image, based on the first 4 characters of the image name
        # Flip dimension order to match gaze data
        label = labels[int(img[:4]) - 1001].transpose(0, 2, 1)
        info = get_gaze_rows(gaze, msg, img)
        if info is None:
            # This is triggered when an image is not present in the eye-tracking data
            # because it was not used as a stimulus.
            continue

        # Extract some metrics to store in the summary DataFrame
        gaze_rows, start_time, end_time = info

        # Before moving on, check how many samples are missing
        consec_invalid, total_invalid = max_consecutive_invalid(gaze_rows)
        fraction_invalid = total_invalid / gaze_rows.shape[0]

        # TODO: Rethink this condition, an alternative:
        #       If fraction is high but consecutive is low, interpolate and keep the sample
        if not (consec_invalid > 60 or fraction_invalid > 0.25):
            # TODO: Consider whether ALL raw gaze data should be saved,
            #       or only the ones that pass the condition above
            # For saving raw gaze, make sure there is a folder 'raw' in 'semPreProc'
            if save_raw_gaze:
                raw_name = os.path.join(
                    base_path,
                    "..",
                    "semPreProc",
                    "raw",
                    f"{participant_number}_{img}.csv",
                )
                gaze_rows.to_csv(raw_name, index=False)

            # Calculate the semantic vector and relative salience, see function description
            sem_vector, dim_presence = get_sem_vector(gaze_rows, label, padding)
            relative_salience = get_relative_salience(sem_vector, dim_presence)

            # Meeting this condition means there was no gaze for the labels in the image
            if max(relative_salience) == 0:
                print("Zero concurrence for image: ", img, " in file: ", et_file[-10:])
                continue

            row = pd.Series(
                {
                    "participant_number": participant_number,
                    "img": img,
                    "start_time": start_time,
                    "end_time": end_time,
                    "consecutive_invalid": consec_invalid,
                    "total_invalid": total_invalid,
                    "sem_vector": sem_vector,
                    "dim_presence": dim_presence,
                    "relative_salience": relative_salience,
                }
            )
            part_summary = pd.concat(
                [part_summary, row.to_frame().T], ignore_index=True
            )
        # If too many invalid samples, print a warning and continue to the next image
        else:
            print(
                "Too many invalid samples in file: ", et_file[-10:], " for image: ", img
            )
            print("consec_invalid: ", consec_invalid, " total_invalid: ", total_invalid)
            continue

    name_string = os.path.join(
        base_path, "..", "semPreProc", f"{participant_number}_padded.csv"
    )
    part_summary.to_csv(name_string, index=False)


if __name__ == "__main__":
    base_path = os.getcwd()

    # Define which folder contains all the eye-tracking data in h5 format
    et_folder = os.path.join(base_path, "..", "osieData")
    et_files = sorted(glob.glob(os.path.join(et_folder, "*.h5")))

    # Define which folder contains all the semantic labels in mat format
    sem_folder = os.path.join(base_path, "..", "osieLabels")
    labels = load_sem_data(sem_folder=sem_folder)

    # Initialise list of image names
    img_names = [f"{i}.jpg" for i in range(1001, 1701)]

    # Loop over files in the made list and extract the eye-tracking data
    for et_file in et_files:
        # TODO: The padding value should depend on the calibration results instead of one value.
        # Some digging needs to be done in the h5 files, which I don't have time for now.
        # The padding value is the radius of the circle around the gaze point that is considered in pixels.
        padding = 5
        extract_et(et_file, img_names, labels, padding, save_raw_gaze=True)
