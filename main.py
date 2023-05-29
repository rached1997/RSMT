import numpy as np
from RSMT.Models.get_data import get_data, get_training_data
from RSMT.Utils.dataset import Dataset
from RSMT.Utils.max_min_generator import MaxMinGenerator
from RSMT.Utils.vector_decoder import VectorDecoder
from RSMT.optimisation_algorithms.genetic_algorithm.geneticalgorithm import geneticalgorithm
from RSMT.optimisation_algorithms.ga_fitness import js_prediction_based_fitness_ga, DSA_fitness_GA
from RSMT.optimisation_algorithms.particle_swarm_optimization.pso import PSO
from RSMT.optimisation_algorithms.psnr import psnr_constraint
from RSMT.optimisation_algorithms.pso_fitness import js_prediction_based_fitness_pso, DSA_fitness_PSO
from RSMT.coverage.coverage import Coverage


def get_data_and_model(dataset_name, model_name, num_data):
    model, whole_data, test_X, test_y, test_indices = get_data(model_name, dataset_name)
    get_sample((test_X, test_y, test_indices), num_data, model)
    return model, whole_data, test_X, test_y, test_indices


def get_sample(testing_dataset, n_elements, main_model):
    x_test, y_test, indices_test = testing_dataset

    test_data = Dataset(x_test, y_test, indices_test, shuffle=False)
    sample_features, sample_labels, sample_indices = test_data.get_sample(sample_size=n_elements)

    prep_sample_features = sample_features.reshape(sample_features.shape[0], sample_features.shape[1],
                                                   sample_features.shape[2], sample_features.shape[3], 1)

    bool_test_predictions = np.array([])
    start = 0
    while start < prep_sample_features.shape[0]:
        original_predictions = np.argmax(main_model.predict(prep_sample_features[start:start + 1000]), axis=1)
        booleans = (original_predictions == np.argmax(sample_labels[start:start + 1000], axis=1))
        bool_test_predictions = np.append(bool_test_predictions, booleans)
        start += 1000
    bool_test_predictions = bool_test_predictions.astype(int)
    sample_features = sample_features[bool_test_predictions]
    sample_labels = sample_labels[bool_test_predictions]
    sample_indices = sample_indices[bool_test_predictions]

    return sample_features, sample_labels, sample_indices


def run_pso(fitness_function, lb, ub, kwargs):
    if fitness_function == "SA":
        pso_inst = PSO(DSA_fitness_PSO, lb, ub, swarmsize=10, maxiter=10,
                       f_ieqcons=psnr_constraint, kwargs=kwargs, debug=True, particle_output=True)
        best_trf, best_trf_fitness, p, fp = pso_inst.run()

    elif fitness_function == "JS":
        pso_inst = PSO(js_prediction_based_fitness_pso, lb, ub, swarmsize=3, maxiter=10,
                       f_ieqcons=psnr_constraint, kwargs=kwargs, debug=True, particle_output=True)
        best_trf, best_trf_fitness, p, fp = pso_inst.run()

    return best_trf, best_trf_fitness, p, fp


def run_ga(fitness_function, max_min_generator, kwargs):
    algorithm_param = {'max_num_iteration': 10, 'population_size': 10, 'mutation_probability': 0.1,
                       'elit_ratio': 0.01, 'crossover_probability': 0.5,
                       'parents_portion': 0.3, 'crossover_type': 'uniform',
                       'max_iteration_without_improv': None, 'maxfixgen': 6}

    var_bound, var_type = max_min_generator.format_max_min_vector()
    dimension = var_type.shape[0]

    if fitness_function == "SA":
        ga = geneticalgorithm(function=DSA_fitness_GA,
                              dimension=dimension,
                              kwargs=kwargs, variable_boundaries=var_bound,
                              variable_type_mixed=var_type, function_timeout=1000,
                              algorithm_parameters=algorithm_param, convergence_curve=True)

    elif fitness_function == "JS":
        ga = geneticalgorithm(function=js_prediction_based_fitness_ga, dimension=dimension,
                              kwargs=kwargs, variable_boundaries=var_bound,
                              variable_type_mixed=var_type, function_timeout=1000,
                              algorithm_parameters=algorithm_param, convergence_curve=True)
        pop = ga.run()

    return pop


def main(metaheurstic_algo, fitness_function, model_name, dataset_name):
    np.random.seed(1339)
    n_elements = 8000
    batch_size = 8

    main_model, whole_data, sample_features, sample_labels, sample_indices = get_data_and_model(dataset_name,
                                                                                                model_name, n_elements)
    sample_size = len(sample_indices)

    #  The activation layers indices
    layers = [3, 6, 10, 12, 15, 19, 22, 27, 29]
    if fitness_function == "SA":
        x_train, y_train = get_training_data(model_name, dataset_name)
        patch_cov = Coverage(model=main_model, x_train=x_train, y_train=y_train, layers=layers, model_name=model_name, dataset_name=dataset_name)
        patch_cov.calculate_metrics(sample_features)

    kwargs = {"whole_data": whole_data, "main_model": main_model}

    patch_start_index = 0
    batch_end_index = patch_start_index + batch_size

    for i in range(sample_size // batch_size):
        kwargs['tracker'].logger.info("--------------------------------iteration : " + str(i))
        original_patches = sample_features[patch_start_index:batch_end_index]
        original_targets = sample_labels[patch_start_index:batch_end_index]
        original_indexes = sample_indices[patch_start_index:batch_end_index]

        kwargs["original_patches"] = original_patches
        kwargs["target"] = original_targets
        kwargs["indices"] = original_indexes

        vector_decoder = VectorDecoder(original_patches, original_indexes, kwargs['whole_data'])
        kwargs['vector_decoder'] = vector_decoder

        if fitness_function in ["KMNC", "NBC", "SA"]:
            kwargs["patch_cov"] = patch_cov

        max_min_generator = MaxMinGenerator(original_patches[0].shape, original_indexes[0])
        ub, lb, final_var_type_vector = max_min_generator.generate_max_min_vector()
        if metaheurstic_algo == "PSO":
            best_trf, best_trf_fitness, p, fp = run_pso(fitness_function, lb, ub, kwargs)
        elif metaheurstic_algo == "GA":
            pop = run_ga(fitness_function, max_min_generator, kwargs)

        patch_start_index += batch_size
        if patch_start_index + batch_size > n_elements:
            batch_end_index = n_elements
        else:
            batch_end_index = patch_start_index + batch_size

    kwargs['tracker'].final_ratio_write()


if __name__ == '__main__':
    main("PSO", "JS", "SSRN", "SA")
