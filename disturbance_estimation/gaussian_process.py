import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from config import load_parameters


class GaussianProcess:
    def __init__(self, number_of_states, number_timesteps):

        self.number_of_past_samples_considered = min(
            number_timesteps, load_parameters.load_param_gaussian_process())

        self.number_of_states = number_of_states

        # Store index k and delta x of every state (difference between prediction and actual state)
        # Todo: self.k_array currently not used
        self.k_array = np.zeros((1, number_timesteps), dtype=int)
        self.delta_x_array = np.zeros((number_of_states, number_timesteps))
        self.numbr_measurements = 0

    def add_delta_x(self, index_k, delta_x):
        self.k_array[0, self.numbr_measurements] = index_k
        self.delta_x_array[:, self.numbr_measurements] = delta_x.reshape(-1)
        self.numbr_measurements += 1

    def plot_distribution(self):
        fig, ax = plt.subplots(self.number_of_states)

        fig.suptitle("Gaussian process of each state")

        idx_considered_for_estimation = list(range(
            self.numbr_measurements-self.number_of_past_samples_considered, self.numbr_measurements))
        print(f"{idx_considered_for_estimation=}")

        for i in range(0, self.number_of_states):
            # A gaussian process has to be plotted for every state
            y_train = self.delta_x_array[i,
                                         idx_considered_for_estimation].reshape(-1, 1)
            X_train = np.atleast_2d(
                list(range(0, self.number_of_past_samples_considered))).reshape(-1, 1)

            kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            gaussian_process = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=9)
            gaussian_process.fit(X_train, y_train)
            print(gaussian_process.kernel_)

            X_predict = np.linspace(
                0, self.number_of_past_samples_considered, self.number_of_past_samples_considered*20).reshape(-1, 1)

            mean_prediction, std_prediction = gaussian_process.predict(
                X_predict, return_std=True)

            ax[i].scatter(X_train, y_train, label="Observations")
            ax[i].plot(X_predict, mean_prediction, label="Mean prediction")
            ax[i].fill_between(
                X_predict.ravel(),
                mean_prediction - 1.96 * std_prediction,
                mean_prediction + 1.96 * std_prediction,
                alpha=0.5,
                label=r"95% confidence interval",
            )
            ax[i].legend()
            ax[i].set(xlabel='x', ylabel='f(x)')
            ax[i].set_title(
                "Gaussian process regression on noise-free dataset")

        plot_real_density = False
        return plot_real_density, fig, ax
