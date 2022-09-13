import numpy as np

from lti_system import lti_system
from data_driven_mpc import data_driven_mpc
from data_driven_mpc import data_driven_predictor
from lti_system import disturbance
from config import load_parameters
from disturbance_estimation import gaussian_process, discounted_kernel_density_estimator

def create_system():
    [main_param, lti_system_param] = load_parameters.load_parameters()

    NUMBER_OF_MEASUREMENTS = main_param["number_of_measurements"]

    # gaussian_process/traditional_kde/discounted_kde
    DISTURBANCE_ESTIMATION = main_param["dist_est"]

    # Specify the type of disturbance for each state
    TYPES_OF_DISTURBANCES = lti_system_param["dist"]  # gaussian/uniform/triangular/lognormal


    A_SYSTEM_MATRIX = lti_system_param["A"]
    B_INPUT_MATRIX = lti_system_param["B"]

    X_INITIAL_STATE = lti_system_param["x_0"]

    INPUT_SEQUENCE = main_param["input_seq"]

    state_disturbances = disturbance.Disturbance(TYPES_OF_DISTURBANCES)

    real_system = lti_system.LTISystem(
        x=X_INITIAL_STATE, A=A_SYSTEM_MATRIX, B=B_INPUT_MATRIX, disturbances=state_disturbances)

    return real_system

def create_controller_modules(real_system):

    [main_param, lti_system_param] = load_parameters.load_parameters()

    NUMBER_OF_MEASUREMENTS = main_param["number_of_measurements"]

    # gaussian_process/traditional_kde/discounted_kde
    DISTURBANCE_ESTIMATION = main_param["dist_est"]

    # Specify the type of disturbance for each state
    TYPES_OF_DISTURBANCES = lti_system_param["dist"]  # gaussian/uniform/triangular/lognormal


    A_SYSTEM_MATRIX = lti_system_param["A"]
    B_INPUT_MATRIX = lti_system_param["B"]

    X_INITIAL_STATE = lti_system_param["x_0"]

    INPUT_SEQUENCE = main_param["input_seq"]

    # Create input-state (needed for Hankel matrix in Data Driven MPC Module)
    state_sequence = np.zeros((X_INITIAL_STATE.shape[0],INPUT_SEQUENCE.shape[1]+1))
    state_sequence[:,0] = X_INITIAL_STATE[:,0]
    # Record input-state sequence
    for i in range(INPUT_SEQUENCE.shape[1]):
        state_sequence[:,i+1] = real_system.next_step(INPUT_SEQUENCE[:,i],add_disturbance=False)[:,0]

    dd_predictor = data_driven_predictor.DDPredictor(INPUT_SEQUENCE,state_sequence)
    dd_mpc = data_driven_mpc.DataDrivenMPC(INPUT_SEQUENCE, state_sequence)


    if DISTURBANCE_ESTIMATION == "gaussian_process":
        disturbance_estimator = gaussian_process.GaussianProcess(X_INITIAL_STATE.shape[0],NUMBER_OF_MEASUREMENTS)
    elif DISTURBANCE_ESTIMATION == "discounted_kde":
        disturbance_estimator = discounted_kernel_density_estimator.DiscountedKDE(X_INITIAL_STATE.shape[0],NUMBER_OF_MEASUREMENTS)

    return dd_mpc,dd_predictor,disturbance_estimator