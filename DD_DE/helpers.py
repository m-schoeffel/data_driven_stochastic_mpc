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
    main_param["input_seq"] = np.zeros((input_seq_len, input_seq_dim))
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


def create_hankel_matrix(input_sequence, state_sequence, prediction_horizon):
    n = prediction_horizon

    dim_u = input_sequence.shape[0]
    dim_x = state_sequence.shape[0]

    length_hankel_matrix = input_sequence.shape[1]-n

    hankel_u_shape = (
        dim_u*(n+1), length_hankel_matrix)
    hankel_x_shape = (
        dim_x*(n+1), length_hankel_matrix)
        
    hankel_u = np.zeros(hankel_u_shape)
    hankel_x = np.zeros(hankel_x_shape)

    # Split input and state sequences horizontally to efficiently concatenate them for varying prediction horizons
    input_sequence_splitted = np.hsplit(
        input_sequence, input_sequence.shape[1])
    state_sequence_splitted = np.hsplit(
        state_sequence, state_sequence.shape[1])

    for i in range(0, length_hankel_matrix):
        hankel_u[:, i] = np.concatenate(
            tuple(input_sequence_splitted[idx] for idx in range(i, i+n+1)))[:, 0]
        hankel_x[:, i] = np.concatenate(
            tuple(state_sequence_splitted[idx] for idx in range(i, i+n+1)))[:, 0]

        h_matrix = np.concatenate((hankel_u, hankel_x))
        return h_matrix
