import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity


class TraditionalKDE:
    def __init__(self,number_of_states):

        # Store index k and delta x of every state (difference between prediction and actual state)
        self.k_array=np.zeros((1,1000),dtype=int)
        self.delta_x = np.zeros((number_of_states,1000))
        self.numbr_measurements = 0

    def add_delta_x(self, index_k, delta_x):
        self.k_array[0,self.numbr_measurements] = index_k
        self.delta_x[:,self.numbr_measurements] = delta_x.reshape(-1)
        self.numbr_measurements += 1

    def plot_distribution(self):
        # A disturbance distribution has to be plotted for every state
        X = self.delta_x[0:self.numbr_measurements,
                               1].reshape(-1, 1)

        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X)

        x_visuell = np.linspace(-5, 5, 3000).reshape(-1, 1)
        logprob = kde.score_samples(x_visuell)

        fig, ax = plt.subplots()
        ax.plot(x_visuell, np.exp(logprob))
        # print(self.delta_x[0:self.numbr_measurements, 1])

        plot_real_density = True
        return plot_real_density, fig, ax
