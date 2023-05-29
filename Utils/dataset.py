import numpy as np

"""  Provides useful functions to process the data
"""


class Dataset:

    def __init__(self, features, labels, indices, reshape=False, new_shape=None, shuffle=True, normalization=False):
        self._pos = 0
        self.data_shape = features[0].shape
        if normalization:
            self._features = self.standardize(features)
        else:
            self._features = features
        self._labels = labels
        self._indices = indices
        self._num_examples = self._features.shape[0]
        if reshape:
            self._features = self._features.reshape(new_shape)
        if shuffle:
            self.shuffling()

    def shuffling(self):
        indices = np.arange(0, self._num_examples)  # get all possible indexes
        np.random.shuffle(indices)  # shuffle indexes
        self._features = self._features[indices]
        self._labels = self._labels[indices]
        self._indices = self._indices[indices]

    def get_num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size, shuffle=True):
        if self._pos + batch_size > self._num_examples or self._pos + batch_size > self._num_examples:
            batch = (self._features[self._pos:self._num_examples], self._labels[self._pos:self._num_examples],
                     self._indices[self._pos:self._num_examples])
            if shuffle:
                self.shuffling()
            self._pos = 0
            return batch
        batch = (self._features[self._pos:self._pos + batch_size], self._labels[self._pos:self._pos + batch_size],
                 self._indices[self._pos:self._pos + batch_size])
        self._pos += batch_size
        return batch

    def get_data(self):
        return self._features, self._labels, self._indices

    def get_shape(self):
        return self.data_shape

    def get_sample_old(self, sample_size):
        if sample_size == self._num_examples:
            return self._features, self._labels, self._indices
        starting_indice = np.random.choice(self._num_examples - sample_size)
        sample = (self._features[starting_indice:starting_indice + sample_size],
                  self._labels[starting_indice:starting_indice + sample_size],
                  self._test_data_indices[starting_indice:starting_indice + sample_size])
        return sample

    def get_sample(self, sample_size):
        if sample_size == self._num_examples:
            return self._features, self._labels, self._indices
        sample_indices = np.random.randint(low=0, high=self._num_examples, size=sample_size)
        sample = (self._features[sample_indices],
                  self._labels[sample_indices],
                  self._indices[sample_indices])
        return sample

    def get_one_hot_encoding(self, labels):
        onehot_labels = np.zeros((len(labels), self._labels.max() + 1))
        onehot_labels[np.arange(len(labels)), labels] = 1
        return onehot_labels

    def standardize(self, features):
        axis = (0, 1, 2) if len(features.shape) == 3 else (0, 1)
        mean_pixel = features.mean(axis=axis, keepdims=True)
        std_pixel = features.std(axis=axis, keepdims=True)
        return np.float32((features - mean_pixel) / std_pixel)
