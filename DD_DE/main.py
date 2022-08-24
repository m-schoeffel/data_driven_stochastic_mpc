import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_chunked

from DD_DE import lti_system
from DD_DE import data_driven_predictor
from DD_DE import data_driven_mpc
from DD_DE import disturbance
from DD_DE import helpers
from disturbance_estimator import gaussian_process, traditional_kernel_density_estimator, discounted_kernel_density_estimator


[main_param, lti_system_param] = helpers.load_parameters()

NUMBER_OF_MEASUREMENTS = main_param["number_of_measurements"]

# gaussian_process/traditional_kde/discounted_kde
DISTURBANCE_ESTIMATION = main_param["dist_est"]

# Specify the type of disturbance for each state
TYPES_OF_DISTURBANCES = lti_system_param["dist"]  # gaussian/uniform/triangular/lognormal


A_SYSTEM_MATRIX = lti_system_param["A"]
B_INPUT_MATRIX = lti_system_param["B"]

X_INITIAL_STATE = lti_system_param["x_0"]

INPUT_SEQUENCE = main_param["input_seq"]

def main():


    my_disturbance = disturbance.Disturbance(TYPES_OF_DISTURBANCES)

    my_system = lti_system.LTISystem(
        x=X_INITIAL_STATE, A=A_SYSTEM_MATRIX, B=B_INPUT_MATRIX, disturbances=my_disturbance)

    state_sequence = np.zeros((X_INITIAL_STATE.shape[0],INPUT_SEQUENCE.shape[1]+1))
    state_sequence[:,0] = X_INITIAL_STATE[:,0]
    # Record input-state sequence
    for i in range(INPUT_SEQUENCE.shape[1]):
        state_sequence[:,i+1] = my_system.next_step(INPUT_SEQUENCE[:,i],add_disturbance=False)[:,0]

    my_predictor = data_driven_predictor.DDPredictor(INPUT_SEQUENCE,state_sequence)
    my_mpc = data_driven_mpc.DataDrivenMPC(INPUT_SEQUENCE, state_sequence)


    if DISTURBANCE_ESTIMATION == "gaussian_process":
        disturbance_estimator = gaussian_process.GaussianProcess(X_INITIAL_STATE.shape[0],NUMBER_OF_MEASUREMENTS)
    elif DISTURBANCE_ESTIMATION == "traditional_kde":
        disturbance_estimator = traditional_kernel_density_estimator.TraditionalKDE(X_INITIAL_STATE.shape[0],NUMBER_OF_MEASUREMENTS)
    elif DISTURBANCE_ESTIMATION == "discounted_kde":
        disturbance_estimator = discounted_kernel_density_estimator.DiscountedKDE(X_INITIAL_STATE.shape[0],NUMBER_OF_MEASUREMENTS)

    # Set initial state
    my_system.x = X_INITIAL_STATE
    print(f"initial state: \n{my_system.x}")

    for _ in range(1, NUMBER_OF_MEASUREMENTS):
        # print(f"\n\nk = {my_system.k}:")

        # Todo: Nächste Zeile muss mit MPC ausgetauscht werden
        u = np.random.randint(-5,5,size=(1,2))
        next_u = my_mpc.get_new_u(my_system.x)
        # print(next_u)

        predicted_state = my_predictor.predict_state(my_system.x, next_u)

        # print(f"Predicted state:  {predicted_state}")
        my_system.next_step(next_u,add_disturbance=False)
        print(f"actual state: \n{my_system.x}")

        delta_x = my_system.x - predicted_state

        disturbance_estimator.add_delta_x(my_system.k, delta_x)

    # print(disturbance_estimator.delta_x_array.shape)

    # plot_real_density, fig, ax = disturbance_estimator.plot_distribution()

    # # Only put real density in plot when it makes sense (not for gaussian process)
    # if plot_real_density:
    #     for i, dist_type in enumerate(TYPES_OF_DISTURBANCES):
    #         print(dist_type)
    #         my_disturbance.plot_real_disturbance(ax[i],dist_type)

    # plt.show()


if __name__ == "__main__":
    main()
