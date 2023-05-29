import os
from RSMT.Utils.generate_final_yaml_file import generate_indices_metadata
import yaml
import numpy as np

"""  Generates the MAX, the MIN and the type vector of all the distortions parameters
     Those functions are useful if you use the Genetic Algorithm (GA)
"""
class MaxMinGenerator:

    def __init__(self, shape, patch_index):
        self.n_lines = shape[0]
        self.n_columns = shape[1]
        self.n_bands = shape[2]
        self.patch_index = patch_index
        self.parameters_metadata = self.build_transformation_metadata("parameters")
        self.indices_metadata = generate_indices_metadata(self.parameters_metadata, 'template.yaml', shape)

    @staticmethod
    def build_transformation_metadata(metadata_type):
        path = os.path.dirname(os.path.abspath(__file__))
        file_name = metadata_type + "_metadata.yaml"
        metadata_path = path +"/"+file_name
        with open(metadata_path) as file:
            tr_metadata = yaml.load(file, Loader=yaml.FullLoader)
        return tr_metadata

    def generate_continuous_line_drop_out_min_max_vector(self):
        vector_size = self.indices_metadata['continuous_line_col_drop_out_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_lines = self.indices_metadata['max_lines'] + 1
        max_bands = self.indices_metadata['max_bands'] + max_lines

        max_vector[np.arange(1, max_lines, 2)] = self.n_lines - 1

        max_vector[np.arange(max_lines, max_bands, 2)] = self.n_bands - 1

        return max_vector, min_vector, var_type_vector

    def generate_discontinuous_line_drop_out_min_max_vector(self):
        vector_size = self.indices_metadata['discontinuous_line_col_drop_out_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_lines = self.indices_metadata['max_lines'] + 1
        max_columns = self.indices_metadata['max_columns'] + max_lines
        max_bands = self.indices_metadata['max_bands'] + max_columns

        max_vector[np.arange(1, max_lines, 2)] = self.n_lines - 1

        max_vector[np.arange(max_lines, max_columns, 2)] = self.n_columns - 1

        max_vector[np.arange(max_columns, max_bands, 2)] = self.n_bands - 1

        return max_vector, min_vector, var_type_vector

    def generate_line_stripping_min_max_vector(self):
        vector_size = self.indices_metadata['line_col_stripping_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_lines = self.indices_metadata['max_lines'] + 1

        max_vector[np.arange(1, max_lines, 2)] = self.n_lines - 1

        max_vector[-2] = self.parameters_metadata["max_mean_bound"] / 100.0
        min_vector[-2] = self.parameters_metadata["min_mean_bound"] / 100.0
        var_type_vector[-2] = 'real'
        max_vector[-1] = self.parameters_metadata["max_std_bound"] / 100.0
        min_vector[-1] = self.parameters_metadata["min_std_bound"] / 100.0
        var_type_vector[-1] = 'real'

        return max_vector, min_vector, var_type_vector

    def generate_line_col_transformations_min_max_vector(self):
        max_transformation_type = 1
        min_transformation_type = 0

        continuous_drop_out_max_vector, continuous_drop_out_min_vector, continuous_drop_out_var_type_vector = \
            self.generate_continuous_line_drop_out_min_max_vector()

        discontinuous_drop_out_max_vector, discontinuous_drop_out_min_vector, discontinuous_drop_out_var_type_vector = \
            self.generate_discontinuous_line_drop_out_min_max_vector()

        stripping_max_vector, stripping_min_vector, stripping_var_type_vector = \
            self.generate_line_stripping_min_max_vector()

        max_vector = np.concatenate((max_transformation_type, continuous_drop_out_max_vector,
                                     discontinuous_drop_out_max_vector, stripping_max_vector), axis=None)
        min_vector = np.concatenate((min_transformation_type, continuous_drop_out_min_vector,
                                     discontinuous_drop_out_min_vector, stripping_min_vector), axis=None)
        var_type_vector = np.concatenate(("int", continuous_drop_out_var_type_vector,
                                          discontinuous_drop_out_var_type_vector, stripping_var_type_vector), axis=None)

        return max_vector, min_vector, var_type_vector

    def generate_region_drop_out_min_max_vector(self):
        vector_size = self.indices_metadata['region_drop_out_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_bands = self.indices_metadata['max_bands'] + 1
        max_x, max_width = self.n_lines - 1, self.n_lines - 1
        max_y, max_length = self.n_columns - 1, self.n_columns - 1

        max_vector[np.arange(1, max_bands, 2)] = self.n_bands - 1

        max_vector[-5:-1] = max_x, max_y, max_width, max_length

        return max_vector, min_vector, var_type_vector

    def generate_spectral_band_loss_min_max_vector(self):
        vector_size = self.indices_metadata['spectral_band_loss_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_bands = self.indices_metadata['max_bands_for_sbl'] + 1

        max_vector[np.arange(1, max_bands, 2)] = self.n_bands - 2
        min_vector[np.arange(1, max_bands, 2)] = 1

        return max_vector, min_vector, var_type_vector

    def generate_salt_and_pepper_min_max_vector(self):
        vector_size = self.indices_metadata['salt_pepper_noise_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_noisy_pixels = int(np.ceil((self.parameters_metadata[
                                            'max_percentage_of_salt_pepper_noisy_pixels'] *
                                        self.n_lines * self.n_columns) / 100) * 3 + 1)

        max_vector[np.arange(1, max_noisy_pixels, 3)] = self.n_lines - 1
        max_vector[np.arange(2, max_noisy_pixels, 3)] = self.n_columns - 1
        max_vector[np.arange(3, max_noisy_pixels, 3)] = 0
        min_vector[np.arange(3, max_noisy_pixels, 3)] = -2

        return max_vector, min_vector, var_type_vector

    def generate_spatial_gaussian_noise_min_max_vector(self):
        vector_size = self.indices_metadata['spatial_gaussian_noise_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_vector[np.arange(1, vector_size, 3)] = self.n_lines - 1
        max_vector[np.arange(2, vector_size, 3)] = self.n_columns - 1
        max_vector[np.arange(3, vector_size, 3)] = 1
        var_type_vector[np.arange(3, vector_size, 3)] = "real"

        return max_vector, min_vector, var_type_vector

    def generate_spectral_gaussian_noise_min_max_vector(self):
        vector_size = self.indices_metadata['spectral_gaussian_noise_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_vector[np.arange(1, vector_size, 2)] = self.n_bands - 1
        max_vector[np.arange(2, vector_size, 2)] = 1
        var_type_vector[np.arange(2, vector_size, 2)] = "real"

        return max_vector, min_vector, var_type_vector

    def generate_rotation_min_max_vector(self):
        vector_size = self.indices_metadata['rotation_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        x, y = self.patch_index
        max_angle = self.parameters_metadata['max_rotation_angle']
        min_angle = self.parameters_metadata['min_rotation_angle']

        max_vector[1:] = x, y, max_angle
        min_vector[1:] = x, y, min_angle
        var_type_vector[-1] = "real"

        return max_vector, min_vector, var_type_vector

    def generate_zoom_min_max_vector(self):
        vector_size = self.indices_metadata['zoom_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        x, y = self.patch_index
        max_zoom_factor = self.parameters_metadata['max_zoom_factor']
        min_zoom_factor = self.parameters_metadata['min_zoom_factor']

        max_vector[1:] = x, y, max_zoom_factor
        min_vector[1:] = x, y, min_zoom_factor
        var_type_vector[-1] = 'real'

        return max_vector, min_vector, var_type_vector

    def generate_max_min_vector(self):
        vector_size = self.indices_metadata['vector_size']

        line_col_transformation_max_vector, line_col_transformation_min_vector, line_col_transformation_var_type_vector \
            = self.generate_line_col_transformations_min_max_vector()
        region_drop_out_max_vector, region_drop_out_min_vector, region_drop_out_var_type_vector = \
            self.generate_region_drop_out_min_max_vector()
        spectral_band_loss_max_vector, spectral_band_loss_min_vector, spectral_band_loss_var_type_vector = \
            self.generate_spectral_band_loss_min_max_vector()
        salt_pepper_noise_max_vector, salt_pepper_noise_min_vector, salt_pepper_noise_var_type_vector = \
            self.generate_salt_and_pepper_min_max_vector()
        spatial_gaussian_noise_max_vector, spatial_gaussian_noise_min_vector, spatial_gaussian_noise_var_type_vector = \
            self.generate_spatial_gaussian_noise_min_max_vector()
        spectral_gn_max_vector, spectral_gn_min_vector, spectral_gn_var_type_vector = \
            self.generate_spectral_gaussian_noise_min_max_vector()
        rotation_max_vector, rotation_min_vector, rotation_var_type_vector = self.generate_rotation_min_max_vector()
        zoom_max_vector, zoom_min_vector, zoom_var_type_vector = self.generate_zoom_min_max_vector()

        final_max_vector = np.concatenate((line_col_transformation_max_vector,
                                           region_drop_out_max_vector,
                                           spectral_band_loss_max_vector,
                                           salt_pepper_noise_max_vector,
                                           spatial_gaussian_noise_max_vector,
                                           spectral_gn_max_vector,
                                           rotation_max_vector,
                                           zoom_max_vector), axis=None)

        final_min_vector = np.concatenate((line_col_transformation_min_vector,
                                           region_drop_out_min_vector,
                                           spectral_band_loss_min_vector,
                                           salt_pepper_noise_min_vector,
                                           spatial_gaussian_noise_min_vector,
                                           spectral_gn_min_vector,
                                           rotation_min_vector,
                                           zoom_min_vector), axis=None)

        final_var_type_vector = np.concatenate((line_col_transformation_var_type_vector,
                                                region_drop_out_var_type_vector,
                                                spectral_band_loss_var_type_vector,
                                                salt_pepper_noise_var_type_vector,
                                                spatial_gaussian_noise_var_type_vector,
                                                spectral_gn_var_type_vector,
                                                rotation_var_type_vector,
                                                zoom_var_type_vector), axis=None)

        assert (vector_size == final_max_vector.shape[0]) and (vector_size == final_min_vector.shape[0]), \
            "the shape of the final max or min transformation vector is wrong !!!!"

        return final_max_vector, final_min_vector, final_var_type_vector

    def format_max_min_vector(self):
        min_vector, max_vector, var_type_vector = self.generate_max_min_vector()
        max_min_vector = np.dstack((max_vector, min_vector))

        return max_min_vector.reshape((max_min_vector.shape[1], max_min_vector.shape[2])), var_type_vector.reshape(
            (var_type_vector.shape[0], 1))