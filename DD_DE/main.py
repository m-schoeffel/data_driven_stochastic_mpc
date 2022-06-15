import numpy as np
import matplotlib.pyplot as plt

from DD_DE import lti_system
from DD_DE import data_driven_predictor
from DD_DE import disturbance
from disturbance_estimator import gaussian_process, traditional_kernel_density_estimator

NUMBER_OF_MEASUREMENTS = 5

# gaussian_process/traditional_kde/discounted_kde
DISTURBANCE_ESTIMATION = "traditional_kde"

TYPE_OF_DISTURBANCE = "lognormal"  # gaussian/uniform/triangular/lognormal

A_SYSTEM_MATRIX = np.array([[1,1],[0,1]])
B_INPUT_MATRIX = np.array([[0],[1]])

X_INITIAL_STATE = np.array([[0],[1]])

INPUT_SEQUENCE = np.zeros((1,9))
INPUT_SEQUENCE[:] = np.array([1, -1, 0, 2, 3, -4, 0, -6, 2])


def main():
    my_disturbance = disturbance.Disturbance(TYPE_OF_DISTURBANCE)

    my_system = lti_system.LTISystem(
        x=X_INITIAL_STATE, A=A_SYSTEM_MATRIX, B=B_INPUT_MATRIX, disturbance=my_disturbance)

    state_sequence = np.zeros((X_INITIAL_STATE.shape[0],INPUT_SEQUENCE.shape[1]+1))
    state_sequence[:,0] = X_INITIAL_STATE[:,0]
    # Record input-state sequence
    for i in range(INPUT_SEQUENCE.shape[1]):
        state_sequence[:,i+1] = my_system.next_step(INPUT_SEQUENCE[:,i],add_disturbance=False)[:,0]

    my_predictor = data_driven_predictor.DDPredictor(INPUT_SEQUENCE,state_sequence)

    if DISTURBANCE_ESTIMATION == "gaussian_process":
        disturbance_estimator = gaussian_process.GaussianProcess(X_INITIAL_STATE.shape[0])
    elif DISTURBANCE_ESTIMATION == "traditional_kde":
        disturbance_estimator = traditional_kernel_density_estimator.TraditionalKDE(X_INITIAL_STATE.shape[0])
    elif DISTURBANCE_ESTIMATION == "discounted_kde":
        # Todo: Change to discounted KDE
        disturbance_estimator = traditional_kernel_density_estimator.TraditionalKDE(X_INITIAL_STATE.shape[0])

    print(f"initial state: {my_system.x}")

    for u in range(1, NUMBER_OF_MEASUREMENTS):
        print(f"\n\nk = {my_system.k}:")

        predicted_state = my_predictor.predict_state(my_system.x, u)

        # print(f"Predicted state:  {my_predictor.predict_state(my_system.x,u)}")
        my_system.next_step(u)
        print(f"actual state: {my_system.x}")

        delta_x = my_system.x - predicted_state

        disturbance_estimator.add_delta_x(my_system.k, delta_x)

    plot_real_density, fig, ax = disturbance_estimator.plot_distribution()

    # Only put real density in plot when it makes sense (not for gaussian process)
    if plot_real_density:
        my_disturbance.plot_real_disturbance(ax)

    plt.show()


if __name__ == "__main__":
    main()
