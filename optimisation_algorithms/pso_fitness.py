import sys
from scipy.spatial import distance
from RSMT.coverage.coverage import *

"""  This file defines the fitness functions of PSO
"""


def js_prediction_based_fitness_pso(transformed_patches_over_gen, **kwargs):
    shape = transformed_patches_over_gen.shape
    transformed_inputs = transformed_patches_over_gen.reshape((-1, shape[2], shape[3], shape[4], 1))

    transformed_outputs = kwargs['main_model'].predict(transformed_inputs)
    original_output = kwargs["main_model"].predict(
        kwargs["original_patches"].reshape(kwargs["original_patches"].shape[0], kwargs["original_patches"].shape[1],
                                           kwargs["original_patches"].shape[2], kwargs["original_patches"].shape[3], 1))

    kwargs['tracker'].temp_data['transformed_outputs'] = transformed_outputs

    original_output = np.tile(original_output, (transformed_patches_over_gen.shape[0], 1))
    js_prediction_fitness = (distance.jensenshannon(original_output, transformed_outputs, axis=1)).reshape(
        (shape[0], shape[1]))
    kwargs['tracker'].temp_data['fx'] = js_prediction_fitness.flatten()
    fitness = np.mean(js_prediction_fitness, axis=1)
    return - fitness


def prediction_based_fitness_pso(transformed_patches_over_gen, **kwargs):
    transformed_inputs = transformed_patches_over_gen.reshape((-1, transformed_patches_over_gen.shape[2],
                                                               transformed_patches_over_gen.shape[3],
                                                               transformed_patches_over_gen.shape[4], 1))

    transformed_outputs = kwargs['main_model'].predict(transformed_inputs)

    kwargs['tracker'].temp_data['transformed_outputs'] = transformed_outputs

    epsilon = sys.float_info.epsilon

    prediction_fitness = np.log(transformed_outputs[:, kwargs["target"].argmax()] + epsilon)

    kwargs['tracker'].temp_data['fx'] = prediction_fitness
    prediction_fitness = prediction_fitness.reshape(len(transformed_patches_over_gen), -1).mean(1)

    return prediction_fitness


def DSA_fitness_PSO(transformed_patches, **kwargs):
    shape = transformed_patches.shape
    transformed_patches = transformed_patches.reshape((shape[0] * shape[1], shape[2], shape[3], shape[4]))
    sections = kwargs["patch_cov"].SA(transformed_patches) - 1
    dsc = sections.reshape(shape[0], shape[1])
    dsc = np.sort(dsc, axis=1)
    mask = (dsc[:, 1:] != dsc[:, :-1])
    mask = np.concatenate((np.full((mask.shape[0], 1), True), mask), axis=1)
    activated_sections = (np.logical_not(kwargs["patch_cov"].buckets[sections]))
    new_activated_sections = np.multiply(activated_sections.reshape(mask.shape), mask)
    new_activated_sections = np.sum(new_activated_sections, axis=1)
    fitness = 1000 - (np.sum(kwargs["patch_cov"].buckets) + new_activated_sections)

    kwargs["patch_cov"].buckets[sections[kwargs['tracker'].temp_data['all_psnr'] >= 20]] = 1

    kwargs['tracker'].temp_data['fx'] = activated_sections

    return fitness
