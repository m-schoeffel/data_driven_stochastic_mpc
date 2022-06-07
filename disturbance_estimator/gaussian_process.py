import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


class GaussianProcess:
    def __init__(self):

        # Store index k and delta x (difference between prediction and actual state)
        self.array_delta_x = np.ones((1000, 2))
        self.numbr_measurements = 0

    def add_delta_x(self, index_k, delta_x):
        self.array_delta_x[self.numbr_measurements, :] = ([index_k, delta_x])
        self.numbr_measurements += 1

    def plot_distribution(self):

        X_train = self.array_delta_x[0:self.numbr_measurements,
                                     0].reshape(-1, 1)
        y_train = self.array_delta_x[0:self.numbr_measurements, 1]

        kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        gaussian_process = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=9)
        gaussian_process.fit(X_train, y_train)
        print(gaussian_process.kernel_)

        X_predict = np.linspace(1, len(X_train)+1, self.numbr_measurements*2).reshape(-1, 1)

        mean_prediction, std_prediction = gaussian_process.predict(
            X_predict, return_std=True)

        fig, ax = plt.subplots()
        ax.scatter(X_train, y_train, label="Observations")
        ax.plot(X_predict, mean_prediction, label="Mean prediction")
        ax.fill_between(
            X_predict.ravel(),
            mean_prediction - 1.96 * std_prediction,
            mean_prediction + 1.96 * std_prediction,
            alpha=0.5,
            label=r"95% confidence interval",
        )
        ax.legend()
        ax.set(xlabel='x', ylabel='f(x)')
        ax.set_title("Gaussian process regression on noise-free dataset")

        plot_real_density = False
        return plot_real_density, fig, ax