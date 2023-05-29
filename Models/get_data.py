import os

import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import scipy.io as sio

"""  Returns the training and testing data (the features, the labels and the indexes)
     Returns the whole data which represents a HSI scene
     Returns the model 
"""

def zero_padding_3D(old_matrix, pad_length, pad_depth=0):
    new_matrix = np.lib.pad(old_matrix, ((pad_length, pad_length), (pad_length, pad_length), (pad_depth, pad_depth)),
                            'constant', constant_values=0)
    return new_matrix


def load_data(name):
    path = os.path.dirname(os.path.abspath(__file__))
    data_path = path + "/SSRN/datasets"
    data = None
    if name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'SA/Salinas_corrected.mat'))['salinas_corrected']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'UP/PaviaU.mat'))['paviaU']

    return data


def get_whole_data(name):
    PATCH_LENGTH = 3
    data_loaded = load_data(name)
    data = data_loaded.reshape(np.prod(data_loaded.shape[:2]), np.prod(data_loaded.shape[2:]))

    data = preprocessing.scale(data)

    data_ = data.reshape(data_loaded.shape[0], data_loaded.shape[1], data_loaded.shape[2])
    whole_data = data_
    padded_data = zero_padding_3D(whole_data, PATCH_LENGTH)
    return padded_data


def get_data_and_labels(file_path, with_indices=False):
    dataset = np.load(file_path)
    X = dataset["X"]
    y = dataset["y"]
    if with_indices:
        indices = dataset["indices"]
        return X, y, indices
    else:
        return X, y


def get_model(model_path):
    loaded_model = tf.keras.models.load_model(model_path)
    return loaded_model


def get_data(model_name, dataset):
    global model, whole_data, indices, x_train, y_train, x_test, y_test
    path = os.path.dirname(os.path.abspath(__file__))
    if model_name == "SSRN":
        if dataset == "PU":
            model = get_model(path + "/SSRN/models/PU/SSRN_PU.hdf5")
            whole_data = get_whole_data("PU")
            x_test, y_test, indices = get_data_and_labels(path + "/SSRN/data/PU/test.npz", with_indices=True)

        if dataset == "SA":
            model = get_model(path + "/SSRN/models/SA/SSRN_SA.hdf5")
            whole_data = get_whole_data("SA")
            x_test, y_test, indices = get_data_and_labels(path + "/SSRN/data/SA/test.npz", with_indices=True)
            print()

    elif model_name == "HybridSN":

        if dataset == "PU":
            model = get_model(path + "/hybridSN/models/PU/hybridSN_PU.hdf5")
            whole_data = np.load(path + "/hybridSN/data/PU/whole_data_after_pca.npy")
            x_test, y_test, indices = get_data_and_labels(path + "/hybridSN/data/PU/test.npz", with_indices=True)

        if dataset == "SA":
            model = get_model(path + "/hybridSN/models/SA/hybridSN_SA.hdf5")
            whole_data = np.load(path + "/hybridSN/data/SA/whole_data_after_pca.npy")
            x_test, y_test, indices = get_data_and_labels(path + "/hybridSN/data/SA/test.npz", with_indices=True)

    return model, whole_data, x_test, y_test, indices


def get_training_data(model_name, dataset):
    global x_train, y_train
    path = os.path.dirname(os.path.abspath(__file__))
    if model_name == "SSRN":

        if dataset == "PU":
            x_train, y_train = get_data_and_labels(path + "/SSRN/data/" + dataset + "/train.npz", with_indices=False)

        if dataset == "SA":
            x_train, y_train = get_data_and_labels(path + "/SSRN/data/" + dataset + "/train.npz", with_indices=False)

    elif model_name == "HybridSN":

        if dataset == "PU":
            x_train, y_train = get_data_and_labels(path + "/hybridSN/data/" + dataset + "/train.npz",
                                                   with_indices=False)

        if dataset == "SA":
            x_train, y_train = get_data_and_labels(path + "/hybridSN/data/" + dataset + "/train.npz",
                                                   with_indices=False)

    return x_train, y_train
