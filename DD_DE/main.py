import warnings

import numpy as np
import matplotlib.pyplot as plt

from DD_DE import lti_system
from DD_DE import data_driven_predictor
from disturbance_estimator import gaussian_process, traditional_kernel_density_estimator

NUMBER_OF_MEASUREMENTS = 500

# gaussian_process/traditional_kde/discounted_kde
DISTURBANCE_ESTIMATION = "traditional_kde"

TYPE_OF_DISTURBANCE = "triangular"  # gaussian/uniform/triangular/lognormal

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
        return np.random.lognormal(mean=0.0, sigma=1.0)

    warnings.warn(
        "No proper disturbance specified, check global variable TYPE_OF_DISTURBANCE")
    return 0


def plot_real_disturbance(ax) -> None:
    number_samples = 1000
    x = np.linspace(-5, 5, number_samples).reshape(-1, 1)

    if TYPE_OF_DISTURBANCE == "gaussian":
        mu = 0
        sigma = 1.0
        pdf = 1/(sigma * np.sqrt(2 * np.pi)) * \
            np.exp(- (x - mu)**2 / (2 * sigma**2))
        ax.plot(x, pdf, linewidth=2, color='r')
    elif TYPE_OF_DISTURBANCE == "uniform":
        lower_bound = -0.5
        upper_bound = 0.5
        x_uniform = np.linspace(lower_bound, upper_bound, number_samples)
        pdf = [1/(upper_bound-lower_bound) for _ in x_uniform]
        ax.plot(x_uniform, pdf, linewidth=2, color='r')
    elif TYPE_OF_DISTURBANCE == "triangular":
        left = -2
        mode = 0.5
        right = 1
        x_triangular = np.linspace(left, right, number_samples)
        left_side = lambda x : 2*(x-left)/((right-left)*(mode-left))
        right_side = lambda x : 2*(right-x)/((right-left)*(right-mode))
        pdf = [left_side(x) for x in x_triangular if x>=left and x<=mode]
        pdf = pdf + [right_side(x) for x in x_triangular if x>mode and x<=right]
        ax.plot(x_triangular, pdf, linewidth=2, color='r')
    elif TYPE_OF_DISTURBANCE == "lognormal":
        mu = 0
        sigma = 1.0
        pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) /
               (x * sigma * np.sqrt(2 * np.pi)))
        plt.plot(x, pdf, linewidth=2, color='r')


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

    # print(f"initial state: {my_system.x}")

    for u in range(1, NUMBER_OF_MEASUREMENTS):
        # print(f"\n\nk = {my_system.k}:")

        predicted_state = my_predictor.predict_state(my_system.x, u)

        # print(f"Predicted state:  {my_predictor.predict_state(my_system.x,u)}")
        my_system.next_step(u)
        # print(f"actual state: {my_system.x}")

        delta_x = my_system.x - predicted_state

        disturbance_estimator.add_delta_x(my_system.k, delta_x)

    plot_real_density, fig, ax = disturbance_estimator.plot_distribution()

    # Only put real density in plot when it makes sense (not for gaussian process)
    if plot_real_density:
        plot_real_disturbance(ax)

    plt.show()


if __name__ == "__main__":
    main()
