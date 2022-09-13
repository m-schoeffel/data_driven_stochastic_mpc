import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity


class TraditionalKDE:
    def __init__(self, number_of_states, max_timesteps):

        self.number_of_states = number_of_states

        # Store index k and delta x of every state (difference between prediction and actual state)
        self.k_array = np.zeros((1, max_timesteps), dtype=int)
        self.delta_x_array = np.zeros((number_of_states, max_timesteps))
        self.numbr_measurements = 0

    def add_delta_x(self, index_k, delta_x):
        self.k_array[0, self.numbr_measurements] = index_k
        self.delta_x_array[:, self.numbr_measurements] = delta_x.reshape(-1)
        self.numbr_measurements += 1

    def plot_distribution(self):
        fig, ax = plt.subplots(self.number_of_states)

        fig.suptitle("Distribution of disturbance on each state")

        for i in range(0, self.number_of_states):
            # A disturbance distribution has to be plotted for every state
            X = self.delta_x_array[i, 0:self.numbr_measurements].reshape(-1, 1)

            kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X)

            x_visuell = np.linspace(-5, 5, 3000).reshape(-1, 1)
            logprob = kde.score_samples(x_visuell)

            ax[i].plot(x_visuell, np.exp(logprob))
            ax[i].set_title(f"State {i+1}")
            # print(self.delta_x[0:self.numbr_measurements, 1])

        plot_real_density = True
        return plot_real_density, fig, ax
