# _RSMT  : HSI testing approach_

This project provide the implementation of `Remote Sensing Metamorphic Tester (RSMT)`.
RSMT is a robustness testing approach for Hyperspectral images (HSIs) classifiers.
It generates adversarial examples(AXs) by using population based optimization algorithms.
Those algorithms will search in the metamorphic search-space of naturally occurring 
HSI distortions in order to find wrong test cases that could mislead the model.

# **The architecture of the project :**

- **`coverage :`**
   - _`coverage.py :`_ provides a class to define the necessary parameters for the surprise adquacy coverage
   - _`surprise_adequacy.py :`_ defines the functions used to calculate the surprise adequacy and the surprise adequacy coverage

- **`Models :`** This directory contains the models and the HSI data. 
The two models we used are SSRN (https://github.com/zilongzhong/SSRN) and HybridSN (https://github.com/gokriznastic/HybridSN)
and for the datasets we used the Pavia University(PU) and Salina (SA) (www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes). The 
Note, here you should add the HSI data as defined below
Models directory is organized as follow : 
    
    - _`SSRN`_
        - _`data`_
            - _` PU :`_ In this directory you should add three files 
                - _`test.npz`_ : this file is an npz containing the Pavia University testing data
                             their labels and their coordinates in the original image after being
                             preprocessed in the same way as SSRN
                - _`train.npz`_ : same as test.npz but with the training data
                - _`val.npz`_ : same as test.npz but with the validation data
            - _`SA :`_ Should contain the same files as PU but this time with SA data
        - _`datasets :`_ contain the row data and the ground truth files. 
             - _` UP :`_ In this directory you should add two files 
                            (1) PaviaU.mat and (2)PaviaU_gt.mat
            - _`SA :`_ Should contain the same files as PU but this time with SA data
           
       - _`models :`_ contains the models of PU and SA (check the references mentioned above to get the models)
            - _`PU`_ : here you should add the file SSRN_PU.hdf5 
            - _`SA`_ : add the file SSRN_SA.hdf5 
    - _`hybridSN :`_ same as SSRN
    - _`get_data :`_ provides the functions to read the data and models from the appropriate files
- **`optimisation_algorithms :`** contains the two used optimisation algorithms Particle Swarm Optimization (PSO) and Genetic Algorithm (GA) 
that were inspired of the those references respectively (https://pythonhosted.org/pyswarm/) and (https://pypi.org/project/geneticalgorithm/) but adapted to HSI data.
This file is structured as follow : 
    - _`genetic_algorithm`_
        - _`geneticalgorithm.py`_
    - _`particle_swarm_optimization`_
        - _`pso.py`_
    - _`ga_fitness.py :`_ contains GA's fitness functions : Jensen-Shanon divergence (JS) and Surprise Adequecy Coverage (DSA)
    - _`pso_fitness.py :`_ defines the fitness function JS and DSA used by the PSO optimizer
    - _`psnr.py :`_ contains the functions to calculate the PSNR
- **`optimisation_algorithms :`**
    - _`transformer_vect.py : `_ contains the implementation of the distortions to be applied on one or more 3D patch
- **`Utils :`**
    - _`dataset.py :`_ class containing useful functions to process the data
    - _`generate_final_yaml_file.py : `_ it generates a file entitled _indices_metadata.yaml_ that contain the index of every distortion in the vector
    it uses the two files _parameters_metadata.yaml_ and _template.yaml_ that defines respectively the boundaries of the distortions parameters and the template to follow to generate the indices file
    -_`max_min_generator :`_ generates vectors defining the upper bound, the lower bound, and the type of all the vector elements (we need this in GA) 
    -_`vector_encoder.py :`_ contains functions to generate the distortions vector
    -_`vector_decoder.py :`_ contains functions to decode the distortions vector and apply the transformations
- **`tracker.py :`**  defines how the data to be tracked
- **`main.py :`** it's the main function, you could change the dataset, the model, the optimisation algorithm and the fitness function as you want
- **`requirement.txt :`** contains the used packages and their versions

# **How to add or modify the distortions  ?**
You need to follow those steps :
1. add the distortion's implementation in the file transformaer_vect.py
2. add how to encode it in a vector (dont miss to add an activation bit at the beginning of the distortion vector) in the file vector_encoder.py
3. define how to decode the distortion vector in vector_decoder.py
4. add in the parameter_metadata.yaml file the boundaries of your distortion
5. in template.yaml define the formula to calculate its index (the formulas follows the jinja2 library requirements)
6. if you wil use GA add how to measure the upper bound, the lower bound and the types of your transformation parameters

