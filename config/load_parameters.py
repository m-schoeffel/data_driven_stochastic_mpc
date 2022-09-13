import numpy as np
import yaml


# Load parameters from config.yaml
# Create dictionaries for each section of parameters
def load_main_params():

    with open('config/config.yaml') as file:
        param = yaml.load(file, Loader=yaml.FullLoader)

    main_param = dict()

    # Two steps necessary, so that input_seq is two dimensional array
    # If dimension of input would be one, input sequence would be one dimensional array otherwise (breaks system)
    input_seq_len = len(param["input_sequence"])
    input_seq_dim = len(param["input_sequence"][0])
    main_param["input_seq"] = np.zeros((input_seq_len, input_seq_dim))
    main_param["input_seq"][:] = np.array(param["input_sequence"])

    main_param["number_of_measurements"] = param["number_of_measurements"]
    main_param["dist_est"] = param["disturbance_estimation"]

    return main_param

def load_lti_system_params():

    with open('config/config.yaml') as file:
        param = yaml.load(file, Loader=yaml.FullLoader)

    lti_system_param = dict()
    lti_system_param["A"] = np.array(param["lti_system"]["a_system_matrix"])
    lti_system_param["B"] = np.array(param["lti_system"]["b_input_matrix"])
    lti_system_param["x_0"] = np.array(param["lti_system"]["x_initial_state"])
    lti_system_param["dist"] = param["lti_system"]["types_of_disturbances"]

    return lti_system_param

def load_cost_matrices():
    with open('config/config.yaml') as file:
        param = yaml.load(file, Loader=yaml.FullLoader)

    cost_matrices = dict()
    cost_matrices["R"] = np.array(param["mpc_cost_matrices"]["R"])
    cost_matrices["Q"] = np.array(param["mpc_cost_matrices"]["Q"])

    return cost_matrices


def load_constraints():
    with open('config/config.yaml') as file:
        param = yaml.load(file, Loader=yaml.FullLoader)

    constraints = dict()
    constraints["G_u"] = np.array(param["constraints"]["G_u"])
    constraints["g_u"] = np.array(param["constraints"]["g_u"])
    constraints["G_x"] = np.array(param["constraints"]["G_x"])
    constraints["g_x"] = np.array(param["constraints"]["g_x"])

    return constraints


def load_prediction_horizon():
    with open('config/config.yaml') as file:
        param = yaml.load(file, Loader=yaml.FullLoader)

    prediction_horizon = np.array(param["prediction_horizon"])

    return prediction_horizon


def load_param_gaussian_process():
    with open('config/config.yaml') as file:
        param = yaml.load(file, Loader=yaml.FullLoader)

    return param["gaussian_process"]["density_number_of_past_samples_considered"]


def load_param_discounted_kde():
    with open('config/config.yaml') as file:
        param = yaml.load(file, Loader=yaml.FullLoader)

    base_of_exponential_weights = param["discounted_kde"]["base_of_exponential_weights"]
    number_of_past_samples_considered = param["discounted_kde"]["density_number_of_past_samples_considered"]

    
    return base_of_exponential_weights, number_of_past_samples_considered
