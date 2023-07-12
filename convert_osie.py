"""
 * @author [Tenzing Dolmans]
 * @email [t.c.dolmans@gmail.com]
 * @create date 2023-05-11 12:06:29
 * @modify date 2023-06-30 17:23:34
 * @desc [description]
"""
import os
import glob
import numpy as np
import pandas as pd
from utils import load_sem_data


def get_gaze_rows(gaze, msg, img):
    """
    Finds the gaze rows that correspond to the start and end times of the image.
    Input:
        gaze: gaze data from the .h5 file
        msg: msg data from the .h5 file
        img: image name
    """
    # Start time when "onset_img_name" and end time when "offset_img_name" in 'msg' matches
    start_time_idx = msg.loc[msg['msg'].str.contains(
        f'onset.*{img}')].first_valid_index()
    end_time_idx = msg.loc[msg['msg'].str.contains(
        f'offset.*{img}')].first_valid_index()

    # Check whether the start and end times are empty, not every image is used as a stimulus
    if start_time_idx is None or end_time_idx is None:
        return None

    start_time = msg.loc[start_time_idx, 'system_time_stamp']
    end_time = msg.loc[end_time_idx, 'system_time_stamp']

    # In 'gaze', find the corresponding rows for the start and end times
    gaze_rows = gaze.loc[(gaze['system_time_stamp'] >= start_time)
                         & (gaze['system_time_stamp'] <= end_time)]
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
    column = gaze_rows['left_gaze_point_valid']
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


def get_sem_vector(gaze_rows, label):
    """
    For a given image and semantic label, calculate the overlap of gaze on semantic label.
    Inputs:
        gaze_rows: DataFrame with gaze data for a given image
        label: Semantic label for the image
    Output:
        sem_vector: A vector of length 12, containing the fraction of gaze for each sem dimension
        dim_presence : A vector of length 12, containing total pixels in each sem dimension
    """
    gaze_data = gaze_rows[['left_gaze_point_on_display_area_x',
                           'left_gaze_point_on_display_area_y',
                           'right_gaze_point_on_display_area_x',
                           'right_gaze_point_on_display_area_y']]

    x = ((gaze_data['left_gaze_point_on_display_area_x'] +
          gaze_data['right_gaze_point_on_display_area_x']) / 2) * 1067
    y = ((gaze_data['left_gaze_point_on_display_area_y'] +
          gaze_data['right_gaze_point_on_display_area_y']) / 2) * 600

    # Lines below fill in 0,0 for NaN values in x and y.
    # TODO: consider the effect of this on the semantic vector in edge cases
    x = list(x.fillna(0).astype(int))
    y = list(y.fillna(0).astype(int))

    # Make sure the values don't exceed the image dimensions (800, 600, so subtract 1)
    x = [min(val, 799) for val in x]
    y = [min(val, 599) for val in x]
    points = np.column_stack((x, y))
    sem_vector = []
    dim_presence = []

    for dim in label:
        matches = np.sum(dim[points[:, 0], points[:, 1]] != 0)
        sem_vector.append(matches / len(gaze_rows))
        dim_presence.append(sum(dim.flatten()))
    return sem_vector, dim_presence


def get_relative_salience(sem_vector, dim_presence):
    """
    Takes the sem_vector and divides it by the dim_presence vector.
    scaled_dim_presence = fraction of pixels in each sem dimension.
    The division amplifies the saliency per dimension if the presence is small:
    large fraction of gaze & small fraction of pixels = high saliency
    Inputs:
        sem_vector: A vector containing the fraction of gaze for each sem dimension
        dim_presence : A vector containing total pixels in each sem dimension
    Output:
        relative_salience: A vector containing the relative saliency for each sem dimension
    """
    max_dim_presence = 800 * 600  # Maximum possible value in dim_presence
    scaled_dim_presence = np.array(dim_presence) / max_dim_presence

    if max(sem_vector) == 0:
        return np.zeros(len(sem_vector))

    relative_salience = np.divide(sem_vector, scaled_dim_presence,
                                  out=np.zeros_like(sem_vector),
                                  where=scaled_dim_presence != 0)
    return list(relative_salience)


def extract_et(et_file, img_names, labels):
    """
    Extracts the eye-tracking data for a given participant.
    Inputs:
        et_file: Path to the eye-tracking file
        img_names: List of image names
        labels: List of semantic labels
    Output:
        part_summary: DataFrame containing the summary data for the participant
        participant_number: DataFrame containing the gaze data for the participant
    """
    part_summary = pd.DataFrame(columns=['participant_number',
                                         'img', 'start_time', 'end_time',
                                         'consecutive_invalid', 'total_invalid',
                                         'sem_vector', 'dim_presence', 'relative_salience'])
    gaze = pd.read_hdf(et_file, 'gaze')
    msg = pd.read_hdf(et_file, 'msg')
    participant_number = os.path.splitext(os.path.basename(et_file))[0]
    print("Processing file: ", participant_number)

    for img in img_names:
        # Select correct label for image, based on the first 4 characters of the image name
        # Flip dimension order to match gaze data
        label = labels[int(img[:4]) - 1001].transpose(0, 2, 1)
        info = get_gaze_rows(gaze, msg, img)
        if info is None:
            # This is triggered when an image is not present in the eye-tracking data
            continue

        # Calculate some metrics to store in the summary DataFrame
        gaze_rows, start_time, end_time = info
        consec_invalid, total_invalid = max_consecutive_invalid(gaze_rows)
        fraction_invalid = total_invalid / gaze_rows.shape[0]

        # TODO: Rethink this condition, several alternatives:
        # 1. If fraction is high but concurrent is low, interpolate and keep it
        if not (consec_invalid > 60 or fraction_invalid > 0.25):
            sem_vector, dim_presence = get_sem_vector(gaze_rows, label)
            relative_salience = get_relative_salience(sem_vector, dim_presence)

            if max(relative_salience) == 0:
                print("Zero concurrence for image: ", img, " in file: ", et_file[-10:])
                continue

            row = pd.Series({'participant_number': participant_number,
                             'img': img,
                             'start_time': start_time,
                             'end_time': end_time,
                             'consecutive_invalid': consec_invalid,
                             'total_invalid': total_invalid,
                             'sem_vector': sem_vector,
                             'dim_presence': dim_presence,
                             'relative_salience': relative_salience})
            part_summary = pd.concat([part_summary, row.to_frame().T], ignore_index=True)

        else:
            print("Too many invalid samples in file: ", et_file[-10:], " for image: ", img)
            print("consec_invalid: ", consec_invalid, " total_invalid: ", total_invalid)
            continue

    return part_summary, participant_number


if __name__ == "__main__":
    base_path = os.getcwd()
    et_folder = os.path.join(base_path, 'osieData')
    et_files = sorted(glob.glob(os.path.join(et_folder, '*.h5')))
    img_names = [f"{i}.jpg" for i in range(1001, 1701)]
    labels = load_sem_data()
    for et_file in et_files:
        part_summary, participant_number = extract_et(et_file, img_names, labels)
        name_string = f"semPreProc/{participant_number}.csv"
        part_summary.to_csv(name_string, index=False)
