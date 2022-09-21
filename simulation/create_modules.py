import numpy as np

from lti_system import lti_system
from data_driven_mpc import data_driven_mpc
from lti_system import disturbance
from config import load_parameters
from disturbance_estimation import gaussian_process, discounted_kernel_density_estimator
from constraint_tightening.constraint_tightening import ConstraintTightening


def create_system():
    lti_system_param = load_parameters.load_lti_system_params()

    # Specify the type of disturbance for each state
    # gaussian/uniform/triangular/lognormal
    TYPES_OF_DISTURBANCES = lti_system_param["dist"]

    A_SYSTEM_MATRIX = lti_system_param["A"]
    B_INPUT_MATRIX = lti_system_param["B"]

    X_INITIAL_STATE = lti_system_param["x_0"]

    state_disturbances = disturbance.Disturbance(TYPES_OF_DISTURBANCES)

    real_system = lti_system.LTISystem(
        x=X_INITIAL_STATE, A=A_SYSTEM_MATRIX, B=B_INPUT_MATRIX, disturbances=state_disturbances)

    return real_system


def create_controller_modules(real_system):

    main_param = load_parameters.load_main_params()
    lti_system_param = load_parameters.load_lti_system_params()

    NUMBER_OF_MEASUREMENTS = main_param["number_of_measurements"]
    NUMBER_OF_INPUTS = main_param["number_of_inputs"]

    # gaussian_process/traditional_kde/discounted_kde
    DISTURBANCE_ESTIMATION = main_param["dist_est"]

    X_INITIAL_STATE = lti_system_param["x_0"]

    # Create input sequence (needed to create input-state sequence)
    dim_u = lti_system_param["B"].shape[1]
    number_of_inputs = NUMBER_OF_INPUTS
    input_sequence = np.random.randint(-10,10,[dim_u,number_of_inputs])

    # Create input-state sequence (needed for Hankel matrix in data_driven_mpc module)
    state_sequence = np.zeros(
        (X_INITIAL_STATE.shape[0], input_sequence.shape[1]+1))
    state_sequence[:, 0] = X_INITIAL_STATE[:, 0]
    # Record input-state sequence
    for i in range(input_sequence.shape[1]):
        state_sequence[:, i+1] = real_system.next_step(
            input_sequence[:, i], add_disturbance=False)[:, 0]

    PREDICTION_HORIZON = load_parameters.load_prediction_horizon()
    cost_matrices = load_parameters.load_cost_matrices()
    dd_mpc = data_driven_mpc.DataDrivenMPC(input_sequence, state_sequence,PREDICTION_HORIZON,cost_matrices["R"],cost_matrices["Q"])

    if DISTURBANCE_ESTIMATION == "gaussian_process":
        disturbance_estimator = gaussian_process.GaussianProcess(
            X_INITIAL_STATE.shape[0], NUMBER_OF_MEASUREMENTS)
    elif DISTURBANCE_ESTIMATION == "discounted_kde":
        [BASE_OF_EXPONENTIAL_WEIGHTS,DEFAULT_NUMBER_PAST_SAMPLES] = load_parameters.load_param_discounted_kde()
        disturbance_estimator = discounted_kernel_density_estimator.DiscountedKDE(
            X_INITIAL_STATE.shape[0], NUMBER_OF_MEASUREMENTS,BASE_OF_EXPONENTIAL_WEIGHTS,DEFAULT_NUMBER_PAST_SAMPLES)

    constraints = load_parameters.load_constraints()
    constraint_tightener = ConstraintTightening(constraints["G_u"],constraints["g_u"],constraints["G_x"],constraints["g_x"])

    return dd_mpc, disturbance_estimator, constraint_tightener
