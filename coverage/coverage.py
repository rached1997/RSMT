import os
from RSMT.coverage.surprise_adequacy import SurpriseAdequacyConfig, DSA, get_sc
import numpy as np

"""  This file contains the necessary parameters definition to measure the Surprise adequacy Coverage (SAC)
"""


class Coverage:
    def __init__(self, model=None, x_train=None, y_train=None, layers=None, SA_k=1000,
                 upper=2, model_name=None, dataset_name=None):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.layers = layers
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.upper = upper
        self.SA_k = SA_k
        self.buckets = np.zeros(SA_k)
        self.dsa = self.init_dsa()

    def init_dsa(self):
        parent_dir_path = os.path.dirname(
            os.path.abspath(__file__)) + "/DSA/" + self.model_name + "/ats/" + self.dataset_name
        layer_names = ['conv3d', 'conv3d_1', 'conv3d_2', 'conv2d', 'dense', 'dense_1']
        # layer_names = ["activation"]
        # for i in range(len(self.layers) - 1):
        #     layer_names.append("activation_" + str(i + 1))
        config = SurpriseAdequacyConfig(saved_path=parent_dir_path, is_classification=True, layer_names=layer_names,
                                        ds_name=self.dataset_name, num_classes=self.y_train.shape[1],
                                        min_var_threshold=1e-5,
                                        batch_size=64, model_name=self.model_name)

        # max_workers = None if self.dataset_name == 'IP' else 5
        dsa_batch_size = 100

        dsa = DSA(model=self.model, train_data=self.x_train, config=config, dsa_batch_size=dsa_batch_size,
                  max_workers=1)
        # use_cache here for reading the training ats from npy file
        dsa.prep(use_cache=True)

        return dsa

    def SA(self, X_target, use_cache=False):
        X_target = X_target.reshape(X_target.shape[0], X_target.shape[1], X_target.shape[2], X_target.shape[3], 1)
        target_dsa, dsa_predictions = self.dsa.calc(X_target, "target", use_cache=use_cache)

        lower = np.amin(target_dsa)
        dsc = get_sc(lower, self.upper, self.SA_k, list(target_dsa))

        return dsc

    def calculate_metrics(self, data):
        dsc = self.SA(data, use_cache=True)
        self.buckets[dsc] = 1
