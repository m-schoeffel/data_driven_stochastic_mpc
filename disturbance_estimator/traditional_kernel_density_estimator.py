import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity


class TraditionalKDE:
    def __init__(self):

        # Store index k and delta x (difference between prediction and actual state)
        self.array_delta_x = np.ones((1000, 2))
        self.numbr_measurements = 0

    def add_delta_x(self, index_k, delta_x):
        self.array_delta_x[self.numbr_measurements, :] = ([index_k, delta_x])
        self.numbr_measurements += 1

    def plot_distribution(self):

        X = self.array_delta_x[0:self.numbr_measurements,
                               1].reshape(-1, 1)

        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X)

        x_visuell = np.linspace(-5, 5, 3000).reshape(-1, 1)
        logprob = kde.score_samples(x_visuell)

        plt.plot(x_visuell, np.exp(logprob))
        plt.scatter(self.array_delta_x[0:self.numbr_measurements, 1], np.ones(self.numbr_measurements),linewidths=0.005)
        print(self.array_delta_x[0:self.numbr_measurements, 1])
        plt.show()
