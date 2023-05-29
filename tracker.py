import os
from sys import getsizeof
import logging
import numpy as np


def reshape_data(data, num_repeat, axis):
    new_data = np.expand_dims(data, axis=axis)
    new_data = np.repeat(new_data, num_repeat, axis=axis)
    if data.ndim == 1:
        new_data = new_data.reshape((-1, ))
    elif data.ndim == 2:
        new_data = new_data.reshape((-1, data.shape[1]))

    return new_data


class Tracker:
    def __init__(self, dataset_name, model_name, optimisation_algo, fitness_func, size = 20):
        path = os.path.dirname(os.path.abspath(__file__)) + "/"
        filename = dataset_name + "_" + model_name +  "_" + optimisation_algo + "_" + fitness_func
        self.file_path = path + filename
        # Set up logging
        log_file = self.file_path + "info.log"
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='Date-Time : %(asctime)s -- Level : %(levelname)s -- Line No. -- %(lineno)d -- msg : %('
                                   'message)s', datefmt='%d/%m/%Y %H:%M:%S')
        self.logger = logging
        self.transformation_data = None
        self.ratio_data = None
        self.temp_data = {}
        self.ratio_vector = np.array([])
        self.max_size = size * 1024 * 1024
        self.counter = {'transformation': 1,
                        'ratio': 1}

    def append_trf_data(self, new_data):
        if self.transformation_data is None:
            self.transformation_data = new_data
        else:
            self.transformation_data = np.append(self.transformation_data, new_data, axis=0)

        if getsizeof(self.transformation_data) >= self.max_size:
            self.write_data(self.transformation_data, "transformation", self.counter["transformation"])
            self.counter["transformation"] += 1
            self.transformation_data = None

    def add_vector_ratio(self, indices):
        indices = indices.flatten().reshape(1, -1)
        self.ratio_vector = np.append(indices, self.ratio_vector).reshape((1, -1))
        if self.ratio_data is None:
            self.ratio_data = self.ratio_vector
        else:
            if self.ratio_data.shape != self.ratio_vector.shape:
                max_columns = max(self.ratio_data.shape[1], self.ratio_vector.shape[1])
                self.ratio_vector = np.pad(self.ratio_vector, ((0, 0), (0, max_columns - self.ratio_vector.shape[1])),
                                           mode='constant', constant_values=-1)
                self.ratio_data = np.pad(self.ratio_data, ((0, 0), (0, max_columns - self.ratio_data.shape[1])),
                                         mode='constant', constant_values=-1)
            self.ratio_data = np.append(self.ratio_data, self.ratio_vector, axis=0)

        self.reset_ratio_vector()
        # if getsizeof(self.ratio_data) >= self.max_size:
        if len(self.ratio_data) >= 20:
            self.write_data(self.ratio_data, "ratio", self.counter["ratio"])
            self.counter["ratio"] += 1
            self.ratio_data = None

    def write_data(self, data, data_type, counter):
        file_path = self.file_path + '_' + data_type + '_' + str(counter) + '.npy'
        np.save(file_path, data)

    def final_trf_write(self):
        self.write_data(self.transformation_data, "transformation", self.counter["transformation"])

    def final_ratio_write(self):
        self.write_data(self.ratio_data, "ratio", self.counter["ratio"])

    def reset_ratio_vector(self):
        self.ratio_vector = np.array([])

    # def read_data(self):
    #     return np.load(self.file_path)

    @staticmethod
    def format_trf_data(fx, x, iteration, indices, psnrs, adverserial_indices, original_softmax):
        batch_size = len(indices)
        pop_size = len(x)
        fs = psnrs >= 20
        x = reshape_data(x, batch_size, axis=1)
        indices = reshape_data(indices, pop_size, axis=0)

        valid_adverserial_indices = np.logical_and(fs, adverserial_indices)
        ids_batch = np.arange(len(valid_adverserial_indices))[valid_adverserial_indices].reshape((-1, 1)) % batch_size
        ids_pop = np.arange(len(valid_adverserial_indices))[valid_adverserial_indices].reshape((-1, 1)) // batch_size
        valid_adverserial_trf = x[valid_adverserial_indices]
        valid_fx = fx[valid_adverserial_indices].reshape((-1, 1))
        valid_indices = indices[valid_adverserial_indices].reshape((-1, 2))
        valid_psnrs = psnrs[valid_adverserial_indices].reshape((-1, 1))
        valid_original_softmax = original_softmax[valid_adverserial_indices]
        iteration = np.array(iteration).repeat(len(valid_adverserial_trf), axis=0).reshape(-1, 1)
        data = np.concatenate((ids_pop, ids_batch, valid_fx, valid_psnrs, iteration, valid_indices,
                               valid_original_softmax, valid_adverserial_trf), axis=1)
        return data

    def format_ratio_data(self, psnr, fs, fg, adverserial_indices):
        all_fs = psnr >= 20
        all_valid_psnr_ratio = np.sum(all_fs) / len(all_fs)
        valid_psnr_ratio = np.sum(fs) / len(fs)
        adverserial_ratio = np.sum(adverserial_indices) / len(adverserial_indices)
        valid_trf_ratio = np.sum(np.logical_and(all_fs, adverserial_indices)) / len(all_fs)
        self.ratio_vector = np.append(self.ratio_vector, np.array([fg, valid_psnr_ratio, all_valid_psnr_ratio,
                                                                   adverserial_ratio, valid_trf_ratio]))

        self.logger.info('Valid psnr ratio : ' + str(all_valid_psnr_ratio))
        self.logger.info('All Valid psnr ratio : ' + str(valid_psnr_ratio))
        self.logger.info('Valid adverserial_ratio : ' + str(adverserial_ratio))
        self.logger.info('Valid valid_trf_ratio : ' + str(valid_trf_ratio))
        self.logger.info('Best Fitness Value : ' + str(fg))
