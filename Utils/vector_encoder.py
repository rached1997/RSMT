import os
import yaml
import numpy as np
import math
from RSMT.Utils.generate_final_yaml_file import generate_indices_metadata

"""  Vector encoder is used to generate a random distortions vector
"""

def generate_activation_value():
    return np.random.randint(low=0, high=2)


def generate_random_percentage(max_percentage=100.0, min_percentage=0.0):
    percentage = np.random.uniform(min_percentage / 100.0, max_percentage / 100.0)
    return percentage


def generate_noise_indices(max_elements, number_noisy):
    indices = np.random.choice(max_elements, size=number_noisy, replace=False)
    return indices


def generate_activated_noisy_elements(number_elements, max_percentage, min_percentage):
    number_noisy_lines = math.ceil(number_elements * generate_random_percentage(
        max_percentage=max_percentage,
        min_percentage=min_percentage))
    activated_lines_indices = generate_noise_indices(max_elements=number_elements,
                                                     number_noisy=number_noisy_lines)
    return number_noisy_lines, activated_lines_indices


def generate_random_binary_values():
    return np.random.choice([0, 1])


def get_random_salt_pepper_indices(pepper_xs, pepper_ys, salt_xs, salt_ys, max_noisy_pixels):
    selected_indices = np.random.choice(np.arange(len(pepper_xs) + len(salt_xs)), int(max_noisy_pixels),
                                        replace=False)

    pepper_indices = selected_indices[selected_indices < len(pepper_xs)]
    salt_indices = selected_indices[selected_indices >= len(pepper_xs)] - len(pepper_xs)
    pepper_xs = pepper_xs[pepper_indices]
    pepper_ys = pepper_ys[pepper_indices]
    salt_xs = salt_xs[salt_indices]
    salt_ys = salt_ys[salt_indices]

    return pepper_xs, pepper_ys, salt_xs, salt_ys


class VectorEncoder(object):

    def __init__(self, shape, patch_index):
        self.n_lines = shape[0]
        self.n_columns = shape[1]
        self.n_bands = shape[2]
        self.patch_index = patch_index
        self.parameters_metadata = self.build_transformation_metadata("parameters")
        self.indices_metadata = generate_indices_metadata(self.parameters_metadata, "template.yaml", shape)

    @staticmethod
    def build_transformation_metadata(metadata_type):
        path = os.path.dirname(os.path.abspath(__file__))
        file_name = metadata_type + "_metadata.yaml"
        metadata_path = path+"/"+file_name
        with open(metadata_path) as file:
            tr_metadata = yaml.load(file, Loader=yaml.FullLoader)
        return tr_metadata

    def generate_continuous_line_col_drop_out_vector(self):
        vector_size = self.indices_metadata['continuous_line_col_drop_out_size']
        output_vector = np.zeros(shape=(vector_size,))

        number_noisy_lines, activated_lines_indices = generate_activated_noisy_elements(
            number_elements=self.n_lines,
            max_percentage=self.parameters_metadata["max_percentage_of_distorted_lines"],
            min_percentage=self.parameters_metadata["min_percentage_of_distorted_lines"])

        number_noisy_bands, activated_bands_indices = generate_activated_noisy_elements(
            number_elements=self.n_bands,
            max_percentage=self.parameters_metadata["max_percentage_of_distorted_bands"],
            min_percentage=self.parameters_metadata["min_percentage_of_distorted_bands"])

        pixel_type = generate_random_binary_values()
        activate_trf = generate_random_binary_values()

        max_lines = self.indices_metadata['max_lines'] + 1

        output_vector[np.arange(1, number_noisy_lines * 2 + 1, 2)] = activated_lines_indices
        output_vector[np.arange(2, number_noisy_lines * 2 + 1, 2)] = 1

        output_vector[np.arange(max_lines, max_lines + number_noisy_bands * 2, 2)] = activated_bands_indices
        output_vector[np.arange(max_lines + 1, max_lines + number_noisy_bands * 2, 2)] = 1

        output_vector[0] = activate_trf
        output_vector[-1] = pixel_type

        return output_vector

    def generate_discontinuous_line_drop_out_vector(self):
        vector_size = self.indices_metadata['discontinuous_line_col_drop_out_size']
        output_vector = np.zeros(shape=(vector_size,))

        number_noisy_lines, activated_lines_indices = generate_activated_noisy_elements(
            number_elements=self.n_lines,
            max_percentage=self.parameters_metadata["max_percentage_of_distorted_lines"],
            min_percentage=self.parameters_metadata["min_percentage_of_distorted_lines"])

        number_noisy_columns, activated_columns_indices = generate_activated_noisy_elements(
            number_elements=self.n_lines,
            max_percentage=self.parameters_metadata["max_percentage_of_distorted_columns"],
            min_percentage=self.parameters_metadata["min_percentage_of_distorted_columns"])

        number_noisy_bands, activated_bands_indices = generate_activated_noisy_elements(
            number_elements=self.n_bands,
            max_percentage=self.parameters_metadata["max_percentage_of_distorted_bands"],
            min_percentage=self.parameters_metadata["min_percentage_of_distorted_bands"])

        pixel_type = generate_random_binary_values()
        activate_trf = generate_random_binary_values()

        max_lines = self.indices_metadata['max_lines'] + 1
        max_lines_columns = self.indices_metadata['max_columns'] + max_lines

        output_vector[np.arange(1, number_noisy_lines * 2 + 1, 2)] = activated_lines_indices
        output_vector[np.arange(2, number_noisy_lines * 2 + 1, 2)] = 1

        output_vector[np.arange(max_lines, max_lines + number_noisy_columns * 2, 2)] = activated_columns_indices
        output_vector[np.arange(max_lines + 1, max_lines + number_noisy_columns * 2, 2)] = 1

        output_vector[np.arange(max_lines_columns, max_lines_columns + number_noisy_bands * 2, 2)] = \
            activated_bands_indices
        output_vector[np.arange(max_lines_columns + 1, max_lines_columns + number_noisy_bands * 2, 2)] = 1

        output_vector[0] = activate_trf
        output_vector[-1] = pixel_type

        return output_vector

    def generate_line_stripping_vector(self):
        vector_size = self.indices_metadata['line_col_stripping_size']
        output_vector = np.zeros(shape=(vector_size,))

        number_noisy_lines, activated_lines_indices = generate_activated_noisy_elements(
            number_elements=self.n_lines,
            max_percentage=self.parameters_metadata["max_percentage_of_distorted_lines"],
            min_percentage=self.parameters_metadata["min_percentage_of_distorted_lines"])

        mean = generate_random_percentage(
            max_percentage=self.parameters_metadata["max_mean_bound"],
            min_percentage=self.parameters_metadata["min_mean_bound"])
        std = generate_random_percentage(
            max_percentage=self.parameters_metadata["max_std_bound"],
            min_percentage=self.parameters_metadata["min_std_bound"])

        activate_trf = generate_random_binary_values()

        output_vector[np.arange(1, number_noisy_lines * 2 + 1, 2)] = activated_lines_indices
        output_vector[np.arange(2, number_noisy_lines * 2 + 1, 2)] = 1

        output_vector[0] = activate_trf
        output_vector[-2] = mean
        output_vector[-1] = std

        return output_vector

    def generate_line_col_transformations_vector(self):
        transformation_type = generate_random_binary_values()
        continuous_drop_out_vector = self.generate_continuous_line_col_drop_out_vector()
        discontinuous_drop_out_vector = self.generate_discontinuous_line_drop_out_vector()
        stripping_vector = self.generate_line_stripping_vector()

        return np.concatenate((transformation_type, continuous_drop_out_vector, discontinuous_drop_out_vector,
                               stripping_vector), axis=None)

    def generate_region_drop_out_vector(self):
        vector_size = self.indices_metadata['region_drop_out_size']
        output_vector = np.zeros(shape=(vector_size,))

        number_noisy_bands, activated_bands_indices = generate_activated_noisy_elements(
            number_elements=self.n_bands,
            max_percentage=self.parameters_metadata["max_percentage_of_distorted_bands"],
            min_percentage=self.parameters_metadata["min_percentage_of_distorted_bands"])

        x, y = (np.random.randint(low=0, high=self.n_lines), np.random.randint(low=0, high=self.n_columns))
        width = np.random.randint(low=0, high=self.n_lines - x)
        length = np.random.randint(low=0, high=self.n_columns - y)
        pixel_type = generate_random_binary_values()
        activate_trf = generate_random_binary_values()

        output_vector[np.arange(1, number_noisy_bands * 2 + 1, 2)] = activated_bands_indices
        output_vector[np.arange(2, number_noisy_bands * 2 + 1, 2)] = 1

        output_vector[0] = activate_trf
        output_vector[-5:] = x, y, width, length, pixel_type

        return output_vector

    def generate_spectral_band_loss_vector(self):
        vector_size = self.indices_metadata['spectral_band_loss_size']
        output_vector = np.zeros(shape=(vector_size,))

        number_noisy_bands, activated_bands_indices = generate_activated_noisy_elements(
            number_elements=self.n_bands,
            max_percentage=self.parameters_metadata["max_percentage_of_distorted_bands_for_sbl"],
            min_percentage=self.parameters_metadata["min_percentage_of_distorted_bands_for_sbl"])

        activated_bands_indices = np.where(activated_bands_indices == 0, 1, activated_bands_indices)
        activated_bands_indices = np.where(activated_bands_indices == self.n_bands - 1, self.n_bands - 2,
                                           activated_bands_indices)
        activate_trf = generate_random_binary_values()

        output_vector[np.arange(1, number_noisy_bands * 2 + 1, 2)] = activated_bands_indices
        output_vector[np.arange(2, number_noisy_bands * 2 + 1, 2)] = 1
        output_vector[0] = activate_trf

        return output_vector

    def generate_salt_and_pepper_vector(self):
        vector_size = self.indices_metadata['salt_pepper_noise_size']
        output_vector = np.zeros(shape=(vector_size,))

        s_vs_p = generate_random_percentage()
        amount = generate_random_percentage(
            min_percentage=self.parameters_metadata['min_percentage_of_salt_pepper_noisy_pixels'],
            max_percentage=self.parameters_metadata['max_percentage_of_salt_pepper_noisy_pixels'])

        # this will maybe generate more than 4 salt and pepper pixels
        mask = np.random.choice([-1, 0, -2], (self.n_lines, self.n_columns),
                                p=[s_vs_p * amount, 1 - (s_vs_p * amount + (1 - s_vs_p) * amount),
                                   (1 - s_vs_p) * amount])

        pepper_xs, pepper_ys = np.where(mask == -1)
        salt_xs, salt_ys = np.where(mask == -2)

        activate_trf = generate_random_binary_values()

        max_noisy_pixels = np.ceil((self.parameters_metadata[
                               'max_percentage_of_salt_pepper_noisy_pixels'] * self.n_lines * self.n_columns) / 100)
        if (len(pepper_xs) + len(salt_xs)) > max_noisy_pixels:
            pepper_xs, pepper_ys, salt_xs, salt_ys = get_random_salt_pepper_indices(pepper_xs, pepper_ys, salt_xs,
                                                                                    salt_ys, max_noisy_pixels)

        output_vector[np.arange(1, len(pepper_xs) * 3 + 1, 3)] = pepper_xs
        output_vector[np.arange(2, len(pepper_ys) * 3 + 1, 3)] = pepper_ys
        output_vector[np.arange(3, len(pepper_ys) * 3 + 1, 3)] = -1

        output_vector[np.arange(len(pepper_ys) * 3 + 1, len(pepper_ys) * 3 + 1 + len(salt_xs) * 3, 3)] = salt_xs
        output_vector[np.arange(len(pepper_xs) * 3 + 2, len(pepper_xs) * 3 + 1 + len(salt_ys) * 3, 3)] = salt_ys
        output_vector[np.arange(len(pepper_xs) * 3 + 3, len(pepper_xs) * 3 + 1 + len(salt_ys) * 3, 3)] = -2

        output_vector[0] = activate_trf

        return output_vector

    def generate_spatial_gaussian_noise_vector(self):
        vector_size = self.indices_metadata['spatial_gaussian_noise_size']
        output_vector = np.zeros(shape=(vector_size,))

        gaussian_p = generate_random_percentage(
            min_percentage=self.parameters_metadata['min_percentage_of_gaussian_noisy_pixels'],
            max_percentage=self.parameters_metadata['max_percentage_of_gaussian_noisy_pixels'])

        mean = generate_random_percentage(
            max_percentage=self.parameters_metadata["max_mean_bound"],
            min_percentage=self.parameters_metadata["min_mean_bound"])
        sigma = generate_random_percentage(
            max_percentage=self.parameters_metadata["max_std_bound"],
            min_percentage=self.parameters_metadata["min_std_bound"])

        mask = np.random.choice([1, 0], (self.n_lines, self.n_columns), p=[gaussian_p, 1 - gaussian_p]) \
            * np.random.normal(mean, sigma, (self.n_lines, self.n_columns))

        gaussian_xs, gaussian_ys = np.where(mask != 0)

        activate_trf = generate_random_binary_values()
        output_vector[0] = activate_trf

        max_noisy_pixels = self.indices_metadata['max_pixels_for_spatial_gn'] / 3

        if len(gaussian_xs) > max_noisy_pixels:
            selected_indices = np.random.choice(np.arange(len(gaussian_xs)), int(max_noisy_pixels),
                                                replace=False)
            gaussian_xs = gaussian_xs[selected_indices]
            gaussian_ys = gaussian_ys[selected_indices]

        output_vector[np.arange(1, len(gaussian_xs) * 3 + 1, 3)] = gaussian_xs
        output_vector[np.arange(2, len(gaussian_xs) * 3 + 1, 3)] = gaussian_ys
        output_vector[np.arange(3, len(gaussian_xs) * 3 + 1, 3)] = mask[gaussian_xs, gaussian_ys]

        return output_vector

    def generate_spectral_gaussian_noise_vector(self):
        vector_size = self.indices_metadata['spectral_gaussian_noise_size']
        output_vector = np.zeros(shape=(vector_size,))

        number_noisy_bands, activated_bands_indices = generate_activated_noisy_elements(
            number_elements=self.n_bands,
            max_percentage=self.parameters_metadata["max_percentage_of_distorted_bands_for_gn"],
            min_percentage=self.parameters_metadata["min_percentage_of_distorted_bands_for_gn"])

        mean = generate_random_percentage(
            max_percentage=self.parameters_metadata["max_mean_bound"],
            min_percentage=self.parameters_metadata["min_mean_bound"])
        sigma = generate_random_percentage(
            max_percentage=self.parameters_metadata["max_std_bound"],
            min_percentage=self.parameters_metadata["min_std_bound"])

        gaussian_noise_values = np.random.normal(mean, sigma, number_noisy_bands)
        activate_trf = generate_random_binary_values()

        output_vector[np.arange(1, number_noisy_bands * 2 + 1, 2)] = activated_bands_indices
        output_vector[np.arange(2, number_noisy_bands * 2 + 1, 2)] = gaussian_noise_values
        output_vector[0] = activate_trf

        return output_vector

    def generate_rotation_vector(self):
        vector_size = self.indices_metadata['rotation_size']
        output_vector = np.zeros(shape=(vector_size,))

        x, y = self.patch_index
        angle = float("{:.3f}".format(np.random.uniform(self.parameters_metadata['min_rotation_angle'],
                                                        self.parameters_metadata['max_rotation_angle'])))
        if angle % 90 == 0:
            angle += np.random.uniform(0.1, 1.0)

        activate_trf = generate_random_binary_values()

        output_vector[:] = activate_trf, x, y, angle

        return output_vector

    def generate_zoom_vector(self):
        vector_size = self.indices_metadata['zoom_size']
        output_vector = np.zeros(shape=(vector_size,))

        x, y = self.patch_index
        zoom_factor = float("{:.3f}".format(np.random.uniform(self.parameters_metadata['min_zoom_factor'],
                                                              self.parameters_metadata['max_zoom_factor'])))
        if zoom_factor == 1:
            zoom_factor += np.random.uniform(0.1, 1.0)

        activate_trf = generate_random_binary_values()

        output_vector[:] = activate_trf, x, y, zoom_factor

        return output_vector

    def construct_random_tr_vector(self):
        vector_size = self.indices_metadata['vector_size']

        line_col_transformation_vector = self.generate_line_col_transformations_vector()
        region_drop_out_vector = self.generate_region_drop_out_vector()
        spectral_band_loss_vector = self.generate_spectral_band_loss_vector()
        salt_pepper_noise_vector = self.generate_salt_and_pepper_vector()
        spatial_gaussian_noise_vector = self.generate_spatial_gaussian_noise_vector()
        spectral_gaussian_noise_vector = self.generate_spectral_gaussian_noise_vector()
        rotation_vector = self.generate_rotation_vector()
        zoom_vector = self.generate_zoom_vector()

        final_vector = np.concatenate((line_col_transformation_vector,
                                       region_drop_out_vector,
                                       spectral_band_loss_vector,
                                       salt_pepper_noise_vector,
                                       spatial_gaussian_noise_vector,
                                       spectral_gaussian_noise_vector,
                                       rotation_vector,
                                       zoom_vector), axis=None)

        assert vector_size == final_vector.shape[0], \
            "the shape of the final transformation vector is wrong !!!!"

        return final_vector
