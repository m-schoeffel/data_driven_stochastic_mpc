import warnings

import numpy as np

from DD_DE import lti_system
from DD_DE import data_driven_predictor
from disturbance_estimator import gaussian_process, traditional_kernel_density_estimator

NUMBER_OF_MEASUREMENTS = 100

# gaussian_process/traditional_kde/discounted_kde
DISTURBANCE_ESTIMATION = "traditional_kde"

TYPE_OF_DISTURBANCE = "gaussian"  # gaussian/uniform/triangular/lognormal

A_SYSTEM_MATRIX = 1
B_INPUT_MATRIX = 1

X_INITIAL_STATE = 0

H_MATRIX = [[1, -1, 0, -1],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, -1, 1, 0]]


def get_dist() -> float:
    if TYPE_OF_DISTURBANCE == "gaussian":
        return np.random.normal(loc=0, scale=1.0)
    elif TYPE_OF_DISTURBANCE == "uniform":
        return np.random.uniform(low=-0.5, high=0.5)
    elif TYPE_OF_DISTURBANCE == "triangular":
        return np.random.triangular(left=-2, mode=0.5, right=1)
    elif TYPE_OF_DISTURBANCE == "lognormal":
        return np.random.lognormal(mean=0.0,sigma=1.0)

    warnings.warn(
        "No proper disturbance specified, check global variable TYPE_OF_DISTURBANCE")
    return 0


def main():
    my_system = lti_system.LTISystem(
        x=X_INITIAL_STATE, A=A_SYSTEM_MATRIX, B=B_INPUT_MATRIX, get_dist=get_dist)

    my_predictor = data_driven_predictor.DDPredictor(H_MATRIX)

    if DISTURBANCE_ESTIMATION == "gaussian_process":
        disturbance_estimator = gaussian_process.GaussianProcess()
    elif DISTURBANCE_ESTIMATION == "traditional_kde":
        disturbance_estimator = traditional_kernel_density_estimator.TraditionalKDE()
    elif DISTURBANCE_ESTIMATION == "discounted_kde":
        # Todo: Change to discounted KDE
        disturbance_estimator = traditional_kernel_density_estimator.TraditionalKDE()

    print(f"actual state: {my_system.x}")

    for u in range(1, NUMBER_OF_MEASUREMENTS):
        # print(f"\n\nk = {my_system.k}:")

        predicted_state = my_predictor.predict_state(my_system.x, u)

        # print(f"Predicted state:  {my_predictor.predict_state(my_system.x,u)}")
        my_system.next_step(u)
        # print(f"actual state: {my_system.x}")

        delta_x = my_system.x - predicted_state

        disturbance_estimator.add_delta_x(my_system.k, delta_x)

    disturbance_estimator.plot_distribution()


if __name__ == "__main__":
    main()
