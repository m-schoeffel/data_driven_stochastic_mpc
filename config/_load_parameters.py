import numpy as np
import yaml


# Load parameters from config.yaml
# Create dictionaries for each section of parameters
def load_main_params():

    with open('config/config.yaml') as file:
        param = yaml.load(file, Loader=yaml.FullLoader)

    main_param = dict()

    main_param["number_of_measurements"] = param["number_of_measurements"]
    main_param["number_of_inputs"] = param["number_of_inputs"]
    main_param["dist_est"] = param["disturbance_estimation"]
    main_param["ref_traj"] = param["reference_trajectory"]
    main_param["dist_seq"] = param["disturbance_sequence"]
    main_param["add_measurement_noise"] = param["add_measurement_noise"]

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

    number_eval_points = param["discounted_kde"]["number_eval_points"]
    interv_min = param["discounted_kde"]["interv_min"]
    interv_max = param["discounted_kde"]["interv_max"]

    return base_of_exponential_weights, number_of_past_samples_considered, number_eval_points, interv_min, interv_max

def load_data_storage_params():
    with open('config/config.yaml') as file:
        param = yaml.load(file, Loader=yaml.FullLoader)

    record_data = param["data_storage"]["record_data"]
    folder_name = param["data_storage"]["folder_name"]

    return record_data, folder_name

def load_risk_param():
    with open('config/config.yaml') as file:
        param = yaml.load(file, Loader=yaml.FullLoader)

    return param["risk_param"]
