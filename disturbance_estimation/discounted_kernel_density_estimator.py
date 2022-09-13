import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from scipy import stats

from data_driven_mpc import helpers


# Todo: Implement interface for disturbance estimators (great exercise)
class DiscountedKDE:
    def __init__(self, number_of_states, number_timesteps):
        [base_of_exponential_weights,default_number_past_samples] = helpers.load_param_discounted_kde()

        # Limit number of consideres samples if not enough samples available
        self.number_of_past_samples_considered_for_kde = min(number_timesteps,default_number_past_samples)

        self.number_of_states = number_of_states

        # Store index k and delta x of every state (difference between prediction and actual state)
        self.k_array = np.zeros((1, number_timesteps), dtype=int)
        self.delta_x_array = np.zeros((number_of_states, number_timesteps))
        self.numbr_measurements = 0

        # Create exponential weight array for discounted kde
        self.weights = np.power(base_of_exponential_weights,np.arange(self.number_of_past_samples_considered_for_kde))
        self.weights = self.weights/np.sum(self.weights)

    def add_delta_x(self, index_k, delta_x):
        self.k_array[0, self.numbr_measurements] = index_k
        self.delta_x_array[:, self.numbr_measurements] = delta_x.reshape(-1)
        self.numbr_measurements += 1

    def plot_distribution(self):
        fig, ax = plt.subplots(self.number_of_states)

        fig.suptitle("Distribution of disturbance on each state (discounted kde)")

        for i in range(0, self.number_of_states):
            # A disturbance distribution has to be plotted for every state
            indices_considered_samples = list(range(self.numbr_measurements-self.number_of_past_samples_considered_for_kde,self.numbr_measurements))
            state_deviations = self.delta_x_array[i, indices_considered_samples]

            print(f"Length weights \n{self.weights.shape}")
            print(f"Length state_deviations \n{state_deviations.shape}")
            # Todo: Implement weights
            kde = stats.gaussian_kde(state_deviations,bw_method=0.1,weights=self.weights)

            x_visuell = np.linspace(-5, 5, 3000)
            prob_distribution = kde.evaluate(x_visuell)

            ax[i].plot(x_visuell, prob_distribution)
            ax[i].set_title(f"State {i+1}")
            # print(self.delta_x[0:self.numbr_measurements, 1])

        plot_real_density = True
        return plot_real_density, fig, ax