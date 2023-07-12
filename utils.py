"""
 * @author [Tenzing Dolmans]
 * @email [t.c.dolmans@gmail.com]
 * @create date 2023-05-11 12:08:55
 * @modify date 2023-05-11 12:08:55
 * @desc [description]
"""
import os
import glob
import torch
import scipy
import numpy as np
from PIL import Image
import scipy.io as sio
import matplotlib.pyplot as plt

def replace_nans(input_tensor, threshold=0.85):

    def pad(data):
        bad_indexes = np.isnan(data)
        good_indexes = np.logical_not(bad_indexes)
        good_data = data[good_indexes]
        if len(good_data) == 0:
            # print("Sample Rejected, no good data")
            return False
        # TODO: refine selection.
        # We need to store the following from input matrix (PyTorch Tensor):
        # - start time
        # - end time
        # - start index
        # - end index
        # - number of samples
        # - number of nans
        # - number of interpolated samples
        # - number of outliers
        # - consecutive number of nans

        elif len(good_data) / len(data) < threshold:
            # print("Sample Rejected: ", len(good_data) / len(data))
            return False
        interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
        data[bad_indexes] = interpolated
        return data
    output = torch.tensor(np.apply_along_axis(pad, 0, input_tensor))
    return output


def downsample(input_tensor, dsf=10):
    """
    Takes input_tensor and downsamples the data by dsf by return a shortened array
    that contains the mode for each of the sample windows.
    """
    ds_t = []
    for i in range(0, len(input_tensor), dsf):
        selection = stats.mode(input_tensor[i: i+dsf], axis=0, keepdims=True)[0][0]
        ds_t.append(torch.tensor(np.array(selection)).unsqueeze(0))
    return torch.cat(ds_t)


def retrieve_semantic_label(file):
    mat = sio.loadmat(file)
    mat = np.array(mat['data'])
    return mat


def load_sem_data(sem_folder='osieLabels', sem_extension='*.mat'):
    """
    Load semantic labels from the OSIE dataset.
    Inputs:
    - sem_folder: folder containing the semantic labels
    Outputs:
    - labels: tensor containing all semantic labels
    """
    sem_paths = sorted(glob.glob(os.path.join(sem_folder, sem_extension)))
    labels = []
    for sem_path in sem_paths:
        # Load semantic labels and convert to tensor
        sem_mat = scipy.io.loadmat(sem_path).get('data')
        labels.append(sem_mat)
    return labels


def load_img_data(img_folder='osieImgs', img_extension='*.jpg'):
    img_paths = sorted(glob.glob(os.path.join(img_folder, img_extension)))
    imgs = []
    for img_path in img_paths:
        # Load image and append to list
        img = Image.open(img_path).convert('RGB').numpy()
        imgs.append(img)
    return imgs

def split_tensor(tensor, sampling_rate=100, selection_length=3):
    """
    Splits every input tensor into multiple usable sections.
    Outputs a composite tensor that contains trainable samples.
    """
    selection_samples = int(sampling_rate * selection_length)
    labels = [data[0] for data in tensor]
    selections = [data[1:, :] for data in tensor]
    data_tensors = []
    for j, selection in enumerate(selections):
        for i in range(0, len(selection) - selection_samples, selection_samples):
            end = i + selection_samples
            _slice = selection[i:end]
            if _slice is not False:
                _labels = labels[j].unsqueeze(0)
                data_tensors.append(torch.cat((_labels, _slice)).unsqueeze(0))
            else:
                print("Selection {} rejected".format(j))
    return torch.cat(data_tensors)


def topk_accuracy(true_labels, pred_labels, k=5):
    """
    Calculate the top k accuracy.
    Inputs:
    - true_labels: list of true labels
    - pred_labels: list of raw output scores from the model
    - k: number of top classes to consider
    Outputs:
    - accuracy: top k accuracy
    """

    # Convert the list of predicted scores to a tensor
    pred_scores = torch.tensor(pred_labels)

    # Compute the softmax over the predicted scores to get probabilities
    pred_probs = torch.softmax(pred_scores, dim=-1)
    # Get the top k classes for each prediction
    _, topk_classes = pred_probs.topk(k, dim=-1)
    # Compute the accuracy for each prediction
    correct = 0
    for i in range(len(true_labels)):
        correct += sum([true_labels[i] in pred for pred in topk_classes[i]])
    topk_accuracy = correct / len(true_labels)

    return topk_accuracy, topk_classes


if __name__ == "__main__":
    current_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..', '..', '..', 'Data', 'GazeBase', 'Data'))
    file_list = list_files_recursively(current_path)
