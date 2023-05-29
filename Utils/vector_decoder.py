import copy
import yaml
import os
import numpy as np
from RSMT.transformations.transformer_vect import Transformer


"""  Vector_decoder is used to  decode the distortion to be applied
"""


def get_activated_element_indices(trf_vector, min_bound, max_bound):
    sub_vector = trf_vector[min_bound: max_bound]
    activated_element_indices = np.array(np.where(sub_vector == 1))
    activated_element_indices = activated_element_indices[activated_element_indices % 2 == 1] - 1
    element_indices = sub_vector[activated_element_indices].astype(int)
    return element_indices


def get_activated_spatial_gaussian_noise_indices(trf_vector):
    gaussian_noise_indices = np.arange(2, len(trf_vector), 3)
    activated_noise = np.where(trf_vector[gaussian_noise_indices] != 0)
    noise_indices = gaussian_noise_indices[activated_noise]
    noise_xs = trf_vector[noise_indices - 2].astype(int)
    noise_ys = trf_vector[noise_indices - 1].astype(int)
    noise_values = trf_vector[noise_indices]

    return noise_xs, noise_ys, noise_values


def get_activated_spectral_gaussian_noise_indices(trf_vector):
    gaussian_noise_indices = np.arange(1, len(trf_vector), 2)
    activated_noise = np.where(trf_vector[gaussian_noise_indices] != 0)
    noise_indices = gaussian_noise_indices[activated_noise]
    noise_zs = trf_vector[noise_indices - 1].astype(int)
    noise_values = trf_vector[noise_indices]

    return noise_zs,  noise_values


def get_salt_pepper_indices(trf_vector, salt_or_pepper):
    noise_indices = np.array(np.where(trf_vector == salt_or_pepper))
    # noise_indices = activated_noise_indices[(activated_noise_indices + 1) % 3 == 0]
    noise_xs_indices = trf_vector[noise_indices - 2].astype(int)
    noise_ys_indices = trf_vector[noise_indices - 1].astype(int)
    return noise_xs_indices, noise_ys_indices


def get_element_from_trf_vector(indices_metadata, transformation_type, index):
    trf_index = indices_metadata[transformation_type + '_index']
    trf_size = indices_metadata[transformation_type + '_size']
    trf_index: trf_size + trf_index
    return trf_index + trf_size + index


def get_gaussian_noise_indices_from_trf_vector(indices_metadata):
    trf_index = indices_metadata['spatial_gaussian_noise_index']
    trf_size = indices_metadata['spatial_gaussian_noise_size']

    spatial_gn_values_indices = np.arange(trf_index + 3, trf_size + trf_index, 3)

    trf_index = indices_metadata['spectral_gaussian_noise_index']
    trf_size = indices_metadata['spectral_gaussian_noise_size']

    spectral_gn_values_indices = np.arange(trf_index + 2, trf_size + trf_index, 2)

    return list(spatial_gn_values_indices) + list(spectral_gn_values_indices)


def format_trf_vector(trf_vector, indices_metadata):
    trf_vector_copy = np.copy(trf_vector)
    mean_index = get_element_from_trf_vector(indices_metadata, "line_col_stripping", -2)
    std_index = get_element_from_trf_vector(indices_metadata, "line_col_stripping", -1)
    angle_index = get_element_from_trf_vector(indices_metadata, "rotation", -1)
    zoom_factor_index = get_element_from_trf_vector(indices_metadata, "zoom", -1)
    gaussian_noise_values_indices = get_gaussian_noise_indices_from_trf_vector(indices_metadata)

    exclude = [mean_index, std_index, angle_index, zoom_factor_index] + gaussian_noise_values_indices

    a = ~np.isin(np.arange(len(trf_vector_copy)), exclude)
    trf_vector_copy[~np.isin(np.arange(len(trf_vector_copy)), exclude)] = np.round(
        trf_vector_copy[~np.isin(np.arange(len(trf_vector_copy)),
                                 exclude)])

    return trf_vector_copy


class VectorDecoder(object):

    def __init__(self, patches, patch_indices, whole_data, trf_vector=None):
        self.whole_data = whole_data
        patch_length = int((patches.shape[1]-1)/2)
        self.transformer = Transformer(copy.deepcopy(whole_data), patch_length)
        self.original_patches = (copy.deepcopy(patches)).reshape(patches.shape[0], patches.shape[1], patches.shape[2], patches.shape[3])
        self.patches = (copy.deepcopy(patches)).reshape(patches.shape[0], patches.shape[1], patches.shape[2], patches.shape[3])
        self.patch_indices = patch_indices
        dead_pixel = np.min(self.original_patches, axis=(1, 2, 3))
        bright_pixel = np.max(self.original_patches, axis=(1, 2, 3))
        self.pixel_values = {"0": dead_pixel, "1": bright_pixel}
        self.indices_metadata = self.build_transformation_metadata("indices")
        if trf_vector is not None:
            self.trf_vector = format_trf_vector(trf_vector, self.indices_metadata)

    def set_trf_vector(self, trf_vector):
        self.trf_vector = format_trf_vector(trf_vector, self.indices_metadata)

    @staticmethod
    def build_transformation_metadata(metadata_type):
        path = os.path.dirname(os.path.abspath(__file__))
        file_name = metadata_type + "_metadata.yaml"
        metadata_path = path + "/" + file_name
        with open(metadata_path) as file:
            tr_metadata = yaml.load(file, Loader=yaml.FullLoader)
        return tr_metadata

    def apply_continuous_line_col_drop_out(self, line_or_col):
        trf_index = self.indices_metadata['continuous_line_col_drop_out_index']
        activation = self.trf_vector[trf_index]
        if activation:
            trf_size = self.indices_metadata['continuous_line_col_drop_out_size']
            trf_vector = self.trf_vector[trf_index + 1: trf_size + trf_index]

            max_lines = self.indices_metadata['max_lines']
            max_bands = self.indices_metadata['max_bands']

            line_indices = get_activated_element_indices(trf_vector, 0, max_lines)
            bands_indices = get_activated_element_indices(trf_vector, max_lines, max_lines + max_bands)
            pixel_values = self.pixel_values[str(int(trf_vector[-1]))]
            if line_or_col:
                self.patches = self.transformer.continuous_line_drop_out(self.patches, line_indices, bands_indices,
                                                                         pixel_values)
            else:
                self.patches = self.transformer.continuous_column_drop_out(self.patches, line_indices, bands_indices,
                                                                           pixel_values)

    def apply_discontinuous_line_col_drop_out(self, line_or_col):
        trf_index = self.indices_metadata['discontinuous_line_col_drop_out_index']
        activation = self.trf_vector[trf_index]
        if activation:
            trf_size = self.indices_metadata['discontinuous_line_col_drop_out_size']
            trf_vector = self.trf_vector[trf_index + 1: trf_size + trf_index]

            max_lines = self.indices_metadata['max_lines']
            max_columns = self.indices_metadata['max_columns']
            max_bands = self.indices_metadata['max_bands']

            line_indices = get_activated_element_indices(trf_vector, 0, max_lines)
            columns_indices = get_activated_element_indices(trf_vector, max_lines, max_lines + max_columns)
            bands_indices = get_activated_element_indices(trf_vector, max_lines + max_columns,
                                                          max_lines + max_columns + max_bands)
            pixel_values = self.pixel_values[str(int(trf_vector[-1]))]

            if line_or_col:
                self.patches = self.transformer.discontinuous_line_drop_out(self.patches, line_indices, columns_indices,
                                                                            bands_indices, pixel_values)
            else:
                self.patches = self.transformer.discontinuous_column_drop_out(self.patches, line_indices,
                                                                              columns_indices, bands_indices,
                                                                              pixel_values)

    def apply_line_stripping(self, line_or_col):
        trf_index = self.indices_metadata['line_col_stripping_index']
        activation = self.trf_vector[trf_index]
        if activation:
            trf_size = self.indices_metadata['line_col_stripping_size']
            trf_vector = self.trf_vector[trf_index + 1: trf_size + trf_index]

            max_lines = self.indices_metadata['max_lines']

            line_indices = get_activated_element_indices(trf_vector, 0, max_lines)
            mean = trf_vector[-2]
            std = trf_vector[-1]

            if line_or_col:
                self.patches = self.transformer.line_stripping(self.patches, line_indices, mean, std)
            else:
                self.patches = self.transformer.column_stripping(self.patches, line_indices, mean, std)

    def apply_line_col_transformations(self):
        line_or_col = self.trf_vector[self.indices_metadata['line_col_transformation_index']]

        self.apply_continuous_line_col_drop_out(line_or_col)
        self.apply_discontinuous_line_col_drop_out(line_or_col)
        self.apply_line_stripping(line_or_col)

    def apply_region_drop_out(self):
        trf_index = self.indices_metadata['region_drop_out_index']
        activation = self.trf_vector[trf_index]
        if activation:
            trf_size = self.indices_metadata['region_drop_out_size']
            trf_vector = self.trf_vector[trf_index + 1: trf_size + trf_index]

            max_bands = self.indices_metadata['max_bands']

            bands_indices = get_activated_element_indices(trf_vector, 0, max_bands)
            x, y, width, length = trf_vector[-5:-1].astype(int)
            pixel_values = self.pixel_values[str(int(trf_vector[-1]))]

            self.patches = self.transformer.region_drop_out(self.patches, (x, y), width, length, bands_indices,
                                                            pixel_values)

    def apply_spectral_band_loss(self):
        trf_index = self.indices_metadata['spectral_band_loss_index']
        activation = self.trf_vector[trf_index]
        if activation:
            trf_size = self.indices_metadata['spectral_band_loss_size']
            trf_vector = self.trf_vector[trf_index + 1: trf_size + trf_index]

            max_bands = self.indices_metadata['max_bands_for_sbl']

            bands_indices = get_activated_element_indices(trf_vector, 0, max_bands)

            self.patches = self.transformer.spectral_band_loss(self.patches, bands_indices)

    def apply_salt_and_pepper(self):
        trf_index = self.indices_metadata['salt_pepper_noise_index']
        activation = self.trf_vector[trf_index]
        if activation:
            trf_size = self.indices_metadata['salt_pepper_noise_size']
            trf_vector = self.trf_vector[trf_index + 1: trf_size + trf_index]

            mask = np.ones((self.patches.shape[0], self.patches.shape[1], self.patches.shape[2]))
            pepper_xs, pepper_ys = get_salt_pepper_indices(trf_vector, -1)
            salt_xs, salt_ys = get_salt_pepper_indices(trf_vector, -2)

            mask[:, pepper_xs, pepper_ys] = np.expand_dims(self.pixel_values['0'], axis=[1, 2])
            mask[:, salt_xs, salt_ys] = np.expand_dims(self.pixel_values['1'], axis=[1, 2])

            self.patches = self.transformer.salt_and_pepper(self.patches, mask)

    def apply_spatial_gaussian_noise(self):
        trf_index = self.indices_metadata['spatial_gaussian_noise_index']
        activation = self.trf_vector[trf_index]
        if activation:
            trf_size = self.indices_metadata['spatial_gaussian_noise_size']
            trf_vector = self.trf_vector[trf_index + 1: trf_size + trf_index]

            mask = np.zeros((self.patches.shape[0], self.patches.shape[1], self.patches.shape[2]))
            gn_xs, gn_ys, noise_values = get_activated_spatial_gaussian_noise_indices(trf_vector)

            mask[:, gn_xs, gn_ys] = noise_values

            self.patches = self.transformer.spatial_gaussian_noise(self.patches, mask)

    def apply_spectral_gaussian_noise(self):
        trf_index = self.indices_metadata['spectral_gaussian_noise_index']
        activation = self.trf_vector[trf_index]
        if activation:
            trf_size = self.indices_metadata['spectral_gaussian_noise_size']
            trf_vector = self.trf_vector[trf_index + 1: trf_size + trf_index]

            bands_indices, noise_values = get_activated_spectral_gaussian_noise_indices(trf_vector)

            self.patches = self.transformer.spectral_gaussian_noise(self.patches, bands_indices, noise_values)

    def apply_rotation(self, i):
        trf_index = self.indices_metadata['rotation_index']
        activation = self.trf_vector[trf_index]
        if activation:
            trf_size = self.indices_metadata['rotation_size']
            trf_vector = self.trf_vector[trf_index + 1: trf_size + trf_index]

            self.patches[i] = self.transformer.rotate(trf_vector[2], (int(self.patch_indices[i][0]),
                                                                      int(self.patch_indices[i][1])))

    def apply_zoom(self, i):
        trf_index = self.indices_metadata['zoom_index']
        activation = self.trf_vector[trf_index]
        if activation:
            trf_size = self.indices_metadata['zoom_size']
            trf_vector = self.trf_vector[trf_index + 1: trf_size + trf_index]

            # try to merge zoom in and zoom out !!!!!!
            if trf_vector[2] > 1:
                self.patches[i] = self.transformer.zoom_in(trf_vector[2], (int(self.patch_indices[i][0]),int(self.patch_indices[i][1])))
            else:
                self.patches[i] = self.transformer.zoom_out(trf_vector[2], (int(self.patch_indices[i][0]),
                                                                            int(self.patch_indices[i][1])))

    def apply_rotations_and_zoom(self):
        for i in range(len(self.patches)):
            self.apply_rotation(i)
            self.apply_zoom(i)

    def apply_rotations_and_zoom_vect(self, i):
        self.apply_rotation(i)
        self.apply_zoom(i)

    def apply_rotations_and_zooms_final(self):
        vectorized_rotations_and_zooms = np.vectorize(self.apply_rotations_and_zoom_vect)
        vectorized_rotations_and_zooms(np.arange(len(self.patches)))

    def apply_transformations(self):
            self.patches = (copy.deepcopy(self.original_patches)).reshape(self.patches.shape)
            self.transformer.set_whole_data(copy.deepcopy(self.whole_data))
            self.apply_rotations_and_zooms_final()
            self.apply_line_col_transformations()
            self.apply_region_drop_out()
            self.apply_spectral_band_loss()
            self.apply_salt_and_pepper()
            self.apply_spatial_gaussian_noise()
            self.apply_spectral_gaussian_noise()

