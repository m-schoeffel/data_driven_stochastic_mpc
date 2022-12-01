import os
import numpy as np
import shutil

from lti_system import _lti_system
from data_driven_mpc import _data_driven_mpc
from lti_system import _disturbance
from config import _load_parameters
from disturbance_estimation import _discounted_kernel_density_estimator, _gaussian_process
from constraint_tightening._constraint_tightening import ConstraintTightening


def create_system():
    lti_system_param = _load_parameters.load_lti_system_params()
    main_param = _load_parameters.load_main_params()

    # Load disturbance_sequence
    dist_seq_name = main_param["dist_seq"]
    path_dist = os.path.join(os.getcwd(),"lti_system","disturbance_sequences",dist_seq_name,dist_seq_name+".npy")
    dist_seq = np.load(path_dist)

    # Specify the type of disturbance for each state
    # gaussian/uniform/triangular/lognormal
    TYPES_OF_DISTURBANCES = lti_system_param["dist"]

    A_SYSTEM_MATRIX = lti_system_param["A"]
    B_INPUT_MATRIX = lti_system_param["B"]

    X_INITIAL_STATE = lti_system_param["x_0"]

    state_disturbances = _disturbance.Disturbance(TYPES_OF_DISTURBANCES)

    real_system = _lti_system.LTISystem(
        x=X_INITIAL_STATE, A=A_SYSTEM_MATRIX, B=B_INPUT_MATRIX, dist_seq=dist_seq)

    return real_system


def create_controller_modules(real_system):

    [record_data, folder_name] = _load_parameters.load_data_storage_params()
    main_param = _load_parameters.load_main_params()
    lti_system_param = _load_parameters.load_lti_system_params()

    # Create folder to store data
    # Program exists with error if folder already exists (important to avoid overriding or modifying datasets)
    if record_data:
        path_root_folder = os.getcwd()
        path = os.path.join(path_root_folder, "recorded_data", folder_name)
        os.mkdir(path)

        # Copy config file to know exact configuration during recording of dataset
        path_config_src = os.path.join(path_root_folder,"config","config.yaml")
        path_config_dest = os.path.join(path,"config.yaml")
        shutil.copy(path_config_src,path_config_dest)

        # Copy reference trajectory
        ref_traj_name = main_param["ref_traj"]
        path_ref_traj_src = os.path.join(path_root_folder,"lti_system","reference_trajectories",ref_traj_name,ref_traj_name+".csv")
        path_ref_traj_dest = os.path.join(path,ref_traj_name+".csv")
        shutil.copy(path_ref_traj_src,path_ref_traj_dest)



    NUMBER_OF_MEASUREMENTS = main_param["number_of_measurements"]
    NUMBER_OF_INPUTS = main_param["number_of_inputs"]

    # gaussian_process/traditional_kde/discounted_kde
    DISTURBANCE_ESTIMATION = main_param["dist_est"]

    X_INITIAL_STATE = lti_system_param["x_0"]

    # Create input sequence (needed to create input-state sequence)
    dim_u = lti_system_param["B"].shape[1]
    number_of_inputs = NUMBER_OF_INPUTS
    input_sequence = np.random.randint(-10, 10, [dim_u, number_of_inputs])

    # Create input-state sequence (needed for Hankel matrix in data_driven_mpc module)
    state_sequence = np.zeros(
        (X_INITIAL_STATE.shape[0], input_sequence.shape[1]+1))
    state_sequence[:, 0] = X_INITIAL_STATE[:, 0]
    # Record input-state sequence
    add_measurement_noise = main_param["add_measurement_noise"]
    for i in range(input_sequence.shape[1]):
        if add_measurement_noise:
            state_sequence[:, i+1] = real_system.next_step(input_sequence[:, i], add_disturbance=False)[:, 0]+np.random.normal(loc=0,scale=0.001,size=4).reshape(-1)
        else:
            state_sequence[:, i+1] = real_system.next_step(input_sequence[:, i], add_disturbance=False)[:, 0]

    PREDICTION_HORIZON = _load_parameters.load_prediction_horizon()
    cost_matrices = _load_parameters.load_cost_matrices()
    dd_mpc = _data_driven_mpc.DataDrivenMPC(
        input_sequence, state_sequence, PREDICTION_HORIZON, cost_matrices["R"], cost_matrices["Q"], record_data, folder_name)

    risk_param = _load_parameters.load_risk_param()

    if DISTURBANCE_ESTIMATION == "gaussian_process":
        disturbance_estimator = _gaussian_process.GaussianProcess(
            X_INITIAL_STATE.shape[0], NUMBER_OF_MEASUREMENTS)
    elif DISTURBANCE_ESTIMATION == "discounted_kde":
        [BASE_OF_EXPONENTIAL_WEIGHTS,
            DEFAULT_NUMBER_PAST_SAMPLES,
            NUMBER_EVAL_POINTS, INTERV_MIN, INTERV_MAX] = _load_parameters.load_param_discounted_kde()
        disturbance_estimator = _discounted_kernel_density_estimator.DiscountedKDE(
            X_INITIAL_STATE.shape[0], NUMBER_OF_MEASUREMENTS, BASE_OF_EXPONENTIAL_WEIGHTS, DEFAULT_NUMBER_PAST_SAMPLES,NUMBER_EVAL_POINTS, INTERV_MIN, INTERV_MAX, record_data, folder_name)

    constraints = _load_parameters.load_constraints()
    constraint_tightener = ConstraintTightening(
        constraints["G_u"], constraints["g_u"], constraints["G_x"], constraints["g_x"], NUMBER_EVAL_POINTS, INTERV_MIN, INTERV_MAX, risk_param, record_data, folder_name)

    return dd_mpc, disturbance_estimator, constraint_tightener
