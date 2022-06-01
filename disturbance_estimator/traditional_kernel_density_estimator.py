import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity


class TraditionalKDE:
    def __init__(self):

        # Store index k and delta x (difference between prediction and actual state)
        self.array_delta_x = np.ones((100, 2))
        self.numbr_measurements = 0

    def add_delta_x(self, index_k, delta_x):
        self.array_delta_x[self.numbr_measurements, :] = ([index_k, delta_x])
        self.numbr_measurements += 1

    def plot_distribution(self):

        X = self.array_delta_x

        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)

        x_visuell = np.linspace(-2, 2, 300).reshape(-1,1)
        y_likelihood = kde.score_samples(x_visuell)

        plt.plot(x_visuell,y_likelihood)
