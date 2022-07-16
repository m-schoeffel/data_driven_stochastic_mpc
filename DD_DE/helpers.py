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

    return main_param, lti_system_param


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


def create_hankel_pseudo_inverse(h_matrix, dim_u, dim_x):

    # Determine prediction horizon of hankel matrix:
    predict_horizon = int(h_matrix.shape[0]/(dim_u+dim_x)-1)

    print(f"\n\npredict_horizon: {predict_horizon}\n")

    # Select relevant rows vor pseudo inverse of hankel matrix (used for prediction step)
    u_rows_idx = list(range(0, dim_u*predict_horizon))
    x_rows_idx = list(range(dim_u*(predict_horizon+1),
                      dim_u*(predict_horizon+1)+dim_x))
    # print(f"u_rows_idx: \n {u_rows_idx}")
    # print(f"x_rows_idx: \n {x_rows_idx}")
    relevant_rows = u_rows_idx + x_rows_idx
    print(relevant_rows)

    h_input_state = h_matrix[relevant_rows, :]
    # print(f"h_input_state (numpy) \n {h_input_state}")
    h_matrix_inv = np.linalg.pinv(h_input_state)

    return h_matrix_inv
