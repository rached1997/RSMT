from jinja2 import Template
import yaml
import copy
import os
import math

"""  Generates a file entitled indices_metadata.yaml that contains the starting index of every distortion.
     Note, this function follows the formulas defined in the file indices_metadata.yaml to generate 
     the indices_metadata.yaml using Jinja2 engine.
"""


def generate_indices_metadata(params, template_path, shape):
    path = os.path.dirname(os.path.abspath(__file__))
    file_path = path + "/indices_metadata.yaml"

    parameters_metadata = copy.deepcopy(params)
    with open(path + "/" + template_path) as file:
        template = yaml.load(file, Loader=yaml.FullLoader)

    n_lines, n_columns, n_bands = shape[0], shape[1], shape[2]
    parameters_metadata['n_lines'] = n_lines
    parameters_metadata['n_columns'] = n_columns
    parameters_metadata['n_bands'] = n_bands

    output = {}
    for k, v in template.items():
        output[k] = math.ceil(float(Template(v).render(parameters_metadata)))
        parameters_metadata[k] = output[k]

    with open(file_path, 'w') as file:
        yaml.dump(output, file, default_flow_style=False)

    return output
