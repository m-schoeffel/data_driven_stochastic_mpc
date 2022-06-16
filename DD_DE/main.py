import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_chunked

from DD_DE import lti_system
from DD_DE import data_driven_predictor
from DD_DE import disturbance
from disturbance_estimator import gaussian_process, traditional_kernel_density_estimator, discounted_kernel_density_estimator

NUMBER_OF_MEASUREMENTS = 100

# gaussian_process/traditional_kde/discounted_kde
DISTURBANCE_ESTIMATION = "discounted_kde"

# Specify the type of disturbance for each state
TYPES_OF_DISTURBANCES = ["lognormal","gaussian"]  # gaussian/uniform/triangular/lognormal

A_SYSTEM_MATRIX = np.array([[1,1],[0,1]])
B_INPUT_MATRIX = np.array([[0],[1]])

X_INITIAL_STATE = np.array([[0],[1]])

INPUT_SEQUENCE = np.zeros((1,9))
INPUT_SEQUENCE[:] = np.array([1, -1, 0, 2, 3, -4, 0, -6, 2])


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

    if DISTURBANCE_ESTIMATION == "gaussian_process":
        disturbance_estimator = gaussian_process.GaussianProcess(X_INITIAL_STATE.shape[0])
    elif DISTURBANCE_ESTIMATION == "traditional_kde":
        disturbance_estimator = traditional_kernel_density_estimator.TraditionalKDE(X_INITIAL_STATE.shape[0],NUMBER_OF_MEASUREMENTS)
    elif DISTURBANCE_ESTIMATION == "discounted_kde":
        # Todo: Change to discounted KDE
        disturbance_estimator = discounted_kernel_density_estimator.DiscountedKDE(X_INITIAL_STATE.shape[0],NUMBER_OF_MEASUREMENTS)

    # print(f"initial sdsatate: {my_system.x}")

    for u in range(1, NUMBER_OF_MEASUREMENTS):
        # print(f"\n\nk = {my_system.k}:")

        predicted_state = my_predictor.predict_state(my_system.x, u)

        # print(f"Predicted state:  {predicted_state}")
        my_system.next_step(u)
        # print(f"actual state: {my_system.x}")

        delta_x = my_system.x - predicted_state

        disturbance_estimator.add_delta_x(my_system.k, delta_x)

    print(disturbance_estimator.delta_x_array.shape)

    plot_real_density, fig, ax = disturbance_estimator.plot_distribution()

    # Only put real density in plot when it makes sense (not for gaussian process)
    if plot_real_density:
        for i, dist_type in enumerate(TYPES_OF_DISTURBANCES):
            print(dist_type)
            my_disturbance.plot_real_disturbance(ax[i],dist_type)

    plt.show()


if __name__ == "__main__":
    main()
