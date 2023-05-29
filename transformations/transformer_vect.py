import numpy as np
from skimage import transform
import sys
import copy
from scipy.ndimage import zoom

"""  This file contains the definition of the distortions.
     The distortions could be applied to one or more patches 
"""


def select_neighboring_patch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1), :]
    selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len + 1)]
    return selected_patch


class Transformer(object):

    def __init__(self, whole_data, patch_length):
        self.whole_data = whole_data
        self.whole_data_copy = copy.deepcopy(self.whole_data)
        self.patch_length = patch_length

    def set_whole_data(self, whole_data):
        self.whole_data = whole_data

    @staticmethod
    def continuous_line_drop_out(transformed_patches, noisy_lines_indices, noisy_bands_indices, pixels_values):
        pixels_values = np.expand_dims(pixels_values, axis=[0, 1, 3])
        pixels_values = np.repeat(pixels_values, len(noisy_lines_indices), axis=0)
        pixels_values = np.repeat(pixels_values, len(noisy_bands_indices), axis=1)
        pixels_values = np.repeat(pixels_values, transformed_patches.shape[2], axis=3)

        transformed_patches[:, np.asarray(noisy_lines_indices)[:, None], :,
        np.asarray(noisy_bands_indices)[None, :]] = pixels_values
        return transformed_patches

    @staticmethod
    def continuous_column_drop_out(transformed_patches, noisy_columns_indices, noisy_bands_indices, pixels_values):
        pixels_values = np.expand_dims(pixels_values, axis=[1, 2, 3])
        pixels_values = np.repeat(pixels_values, transformed_patches.shape[1], axis=1)
        pixels_values = np.repeat(pixels_values, len(noisy_columns_indices), axis=2)
        pixels_values = np.repeat(pixels_values, len(noisy_bands_indices), axis=3)
        transformed_patches[:, :, np.asarray(noisy_columns_indices)[:, None],
        np.asarray(noisy_bands_indices)[None, :]] = pixels_values
        return transformed_patches

    @staticmethod
    def discontinuous_line_drop_out(transformed_patches, noisy_lines_indices, noisy_columns_indices,
                                    noisy_bands_indices, pixels_values):
        pixels_values = np.expand_dims(pixels_values, axis=[1, 2, 3])
        pixels_values = np.repeat(pixels_values, len(noisy_lines_indices), axis=1)
        pixels_values = np.repeat(pixels_values, len(noisy_columns_indices), axis=2)
        pixels_values = np.repeat(pixels_values, len(noisy_bands_indices), axis=3)
        mask_add = np.zeros(transformed_patches.shape)
        mask_mull = np.ones(transformed_patches.shape)
        mask_add[:, noisy_lines_indices[:, None, None], noisy_columns_indices[None, :, None],
        noisy_bands_indices[None, None, :]] = pixels_values
        mask_mull[:, noisy_lines_indices[:, None, None], noisy_columns_indices[None, :, None],
        noisy_bands_indices[None, None, :]] = 0

        return (transformed_patches * mask_mull) + mask_add

    @staticmethod
    def discontinuous_column_drop_out(transformed_patches, noisy_columns_indices, noisy_lines_indices,
                                      noisy_bands_indices, pixels_values):
        pixels_values = np.expand_dims(pixels_values, axis=[1, 2, 3])
        pixels_values = np.repeat(pixels_values, len(noisy_lines_indices), axis=1)
        pixels_values = np.repeat(pixels_values, len(noisy_columns_indices), axis=2)
        pixels_values = np.repeat(pixels_values, len(noisy_bands_indices), axis=3)
        mask_add = np.zeros(transformed_patches.shape)
        mask_mull = np.ones(transformed_patches.shape)
        mask_add[:, noisy_lines_indices[:, None, None], noisy_columns_indices[None, :, None],
        noisy_bands_indices[None, None, :]] = pixels_values
        mask_mull[:, noisy_lines_indices[:, None, None], noisy_columns_indices[None, :, None],
        noisy_bands_indices[None, None, :]] = 0

        return (transformed_patches * mask_mull) + mask_add

    @staticmethod
    def line_stripping(transformed_patches, noisy_lines_indices, mean_noise, std_noise):
        epsilon = sys.float_info.epsilon

        mean = transformed_patches[:, noisy_lines_indices, :, :].mean(2)
        mean = np.repeat(np.expand_dims(mean, axis=2), transformed_patches.shape[2], axis=2)

        std = transformed_patches[:, noisy_lines_indices, :, :].std(axis=2)
        std[np.where(std == 0)[0]] = epsilon
        std = np.repeat(np.expand_dims(std, axis=2), transformed_patches.shape[2], axis=2)

        transformed_patches[:, noisy_lines_indices, :, :] = (std_noise / std) * (
                transformed_patches[:, noisy_lines_indices, :, :] - mean + (std / std_noise) * mean_noise)
        return transformed_patches

    @staticmethod
    def column_stripping(transformed_patches, noisy_columns_indices, mean_noise, std_noise):
        epsilon = sys.float_info.epsilon

        mean = transformed_patches[:, :, noisy_columns_indices, :].mean(1)
        mean = np.repeat(np.expand_dims(mean, axis=1), transformed_patches.shape[1], axis=1)

        std = transformed_patches[:, :, noisy_columns_indices, :].std(axis=1)
        std[np.where(std == 0)[0]] = epsilon
        std = np.repeat(np.expand_dims(std, axis=1), transformed_patches.shape[1], 1)

        transformed_patches[:, :, noisy_columns_indices, :] = (std_noise / std) * (
                transformed_patches[:, :, noisy_columns_indices, :] - mean + (std / std_noise) * mean_noise)
        return transformed_patches

    @staticmethod
    def region_drop_out(transformed_patches, xy, width, length, noisy_bands_indices, pixels_values):
        try:
            if (xy[0] + width) > transformed_patches.shape[1]:
                width = transformed_patches.shape[1] - xy[0]
            if (xy[1] + length) > transformed_patches.shape[2]:
                length = transformed_patches.shape[2] - xy[1]
            pixels_values = np.expand_dims(pixels_values, axis=[1, 2, 3])
            pixels_values = np.repeat(pixels_values, width, axis=1)
            pixels_values = np.repeat(pixels_values, length, axis=2)
            pixels_values = np.repeat(pixels_values, len(noisy_bands_indices), axis=3)
            transformed_patches[:, xy[0]:xy[0] + width, xy[1]:xy[1] + length, noisy_bands_indices] = pixels_values
            return transformed_patches
        except Exception as e:
            print(e)
            return transformed_patches

    @staticmethod
    def spectral_band_loss(transformed_patches, noisy_bands_indices):
        try:
            transformed_patches[:, :, :, noisy_bands_indices] = np.mean([
                transformed_patches[:, :, :, :-2], transformed_patches[:, :, :, 2:]],
                axis=0)[:, :, :, noisy_bands_indices - 1]
            return transformed_patches
        except Exception as e:
            print(e)
            return transformed_patches

    @staticmethod
    def salt_and_pepper(transformed_patches, mask):
        try:
            mask_mul = np.where(mask != 1, 0, mask)
            mask_add = np.where(mask == 1, 0, mask)

            mask_mul = np.repeat(np.expand_dims(mask_mul, axis=3), transformed_patches.shape[3], axis=3)
            mask_add = np.repeat(np.expand_dims(mask_add, axis=3), transformed_patches.shape[3], axis=3)

            return (transformed_patches * mask_mul) + mask_add
        except Exception as e:
            print(e)
            return transformed_patches

    @staticmethod
    def spatial_gaussian_noise(transformed_patches, mask):
        gauss = np.repeat(np.expand_dims(mask, axis=3), transformed_patches.shape[3], axis=3)

        transformed_patches = transformed_patches + gauss
        return transformed_patches

    @staticmethod
    def spectral_gaussian_noise(transformed_patches, noisy_bands_indices, noise_values):
        transformed_patches = transformed_patches.astype('float64')
        transformed_patches[:, :, :, noisy_bands_indices] += noise_values
        return transformed_patches

    def rotate(self, rotation_angle, center):
        rotated_image = transform.rotate(self.whole_data, rotation_angle, resize=False, center=[center[1], center[0]],
                                         preserve_range=True)
        transformed_patch = select_neighboring_patch(rotated_image, center[0], center[1], self.patch_length)
        # save rotated scene
        self.whole_data_copy = rotated_image

        return transformed_patch

    def zoom_out(self, zoom_factor, center):
        h, w = self.whole_data_copy.shape[:2]
        translation = transform.SimilarityTransform(translation=[-((w // 2) - center[1]), -((h // 2) - center[0])])
        img = transform.warp(self.whole_data_copy, translation)

        zoom_tuple = (zoom_factor,) * 2 + (1,)

        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        out = np.zeros_like(img)
        out[top:top + zh, left:left + zw] = zoom(img, zoom_tuple, order=0)
        return select_neighboring_patch(out, out.shape[0] // 2, out.shape[1] // 2, self.patch_length)

    def zoom_in(self, zoom_factor, center):
        h, w = self.whole_data_copy.shape[:2]
        translation = transform.SimilarityTransform(translation=[-((w // 2) - center[1]), -((h // 2) - center[0])])
        img = transform.warp(self.whole_data_copy, translation)
        zoom_tuple = (zoom_factor,) * 2 + (1,)
        zh = int(np.ceil(h / zoom_factor))
        zw = int(np.ceil(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        out = zoom(img[top:top + zh, left:left + zw], zoom_tuple, order=0)
        return select_neighboring_patch(out, out.shape[0] // 2, out.shape[1] // 2, self.patch_length)
