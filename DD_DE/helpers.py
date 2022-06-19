import numpy as np
import yaml


# Load parameters from config.yaml
# Create dictionaries for each section of parameters
def load_parameters():

    with open('config/config.yaml') as file:
        param = yaml.load(file, Loader=yaml.FullLoader)

    lti_system = dict()
    lti_system["A"] = np.array(param["lti_system"]["a_system_matrix"])
    lti_system["B"] = np.array(param["lti_system"]["b_input_matrix"])
    lti_system["x_0"] = np.array(param["lti_system"]["x_initial_state"])
    lti_system["dist"] = param["lti_system"]["types_of_disturbances"]

    print(lti_system)

