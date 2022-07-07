import numpy as np
import yaml


# Load parameters from config.yaml
# Create dictionaries for each section of parameters
def load_parameters():

    with open('config/config.yaml') as file:
        param = yaml.load(file, Loader=yaml.FullLoader)

    main_param = dict()

    # Two steps necessary, so that input_seq is two dimensional array
    # If dimension of input would be one, input sequence would be one dimensional array otherwise (breaks system)
    input_seq_len = len(param["input_sequence"])
    input_seq_dim = len(param["input_sequence"][0])
    main_param["input_seq"] = np.zeros((input_seq_len,input_seq_dim))
    main_param["input_seq"][:] = np.array(param["input_sequence"])

    main_param["number_of_measurements"] = param["number_of_measurements"]
    main_param["dist_est"] = param["disturbance_estimation"]

    lti_system_param = dict()
    lti_system_param["A"] = np.array(param["lti_system"]["a_system_matrix"])
    lti_system_param["B"] = np.array(param["lti_system"]["b_input_matrix"])
    lti_system_param["x_0"] = np.array(param["lti_system"]["x_initial_state"])
    lti_system_param["dist"] = param["lti_system"]["types_of_disturbances"]

    disc_kde_param = dict()
    disc_kde_param["base_exp_weights"] = param["discounted_kde"]["base_of_exponential_weights"]
    disc_kde_param["samples_considered"] = param["discounted_kde"]["number_of_past_samples_considered"]

    return main_param, lti_system_param, disc_kde_param
