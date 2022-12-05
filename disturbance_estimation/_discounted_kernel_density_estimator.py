import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from scipy import stats


# Todo: Implement interface for disturbance estimators (great exercise)
class DiscountedKDE:
    def __init__(self, number_of_states, number_timesteps, base_of_exponential_weights, default_number_past_samples, number_eval_points, interv_min, interv_max, record_data=False, folder_name=""):

        # Limit number of considered samples if not enough samples available
        self.number_of_past_samples_considered_for_kde = min(
            number_timesteps, default_number_past_samples)

        self.number_of_states = number_of_states

        # Store index k and delta x of every state (difference between prediction and actual state)
        # self.k_array = np.zeros((1, number_timesteps), dtype=int)
        # self.numbr_measurements = 0

        # Initialize deque which stores delta_x
        self.delta_x_deque = deque([np.random.normal(0, 0.02, size=[number_of_states]) for _ in range(
            0, self.number_of_past_samples_considered_for_kde)])
        # self.delta_x_deque = deque([0 for _ in range(
        #     0, self.number_of_past_samples_considered_for_kde)])

        # Create exponential weight array for discounted kde
        self.weights = np.power(base_of_exponential_weights, np.arange(
            self.number_of_past_samples_considered_for_kde))
        self.weights = self.weights/np.sum(self.weights)

        self.number_eval_points = number_eval_points
        self.interv_min = interv_min
        self.interv_max = interv_max

        # Create folders to store data
        self.record_data = record_data
        if self.record_data:
            path_root_folder = os.getcwd()
            record_folder_path = os.path.join(path_root_folder, "recorded_data", folder_name, "disturbance_estimation")
            os.mkdir(record_folder_path)
            
            self.dist_x0_path = os.path.join(record_folder_path,"disturbance_distribution_x_0")
            os.mkdir(self.dist_x0_path)

            self.b_coeff_path = os.path.join(record_folder_path,"bhattacharyya_coefficients")
            os.mkdir(self.b_coeff_path)

            self.weights_path = os.path.join(record_folder_path,"weights")
            os.mkdir(self.weights_path)

    def add_delta_x(self, index_k, delta_x):
        # Todo: remove or use k_array
        # self.k_array[0, self.numbr_measurements] = index_k
        # self.numbr_measurements += 1

        self.delta_x_deque.popleft()
        self.delta_x_deque.append(delta_x.reshape(-1))

    def plot_distribution(self):
        fig, ax = plt.subplots(self.number_of_states)

        fig.suptitle(
            "Distribution of disturbance on each state (discounted kde)")

        delta_x_storage = np.zeros(
            [self.number_of_states, self.number_of_past_samples_considered_for_kde])
        for i, delta_x in enumerate(self.delta_x_deque):
            delta_x_storage[:, i] = delta_x

        for i in range(0, self.number_of_states):
            # A disturbance distribution is plotted for every state
            state_deviations = delta_x_storage[i, :]

            # Todo: Implement weights
            kde = stats.gaussian_kde(
                state_deviations, weights=self.weights[i,:])

            x_visuell = np.linspace(-5, 5, 3000)
            prob_distribution = kde.evaluate(x_visuell)

            ax[i].plot(x_visuell, prob_distribution)
            ax[i].set_title(f"State {i+1}")

        plot_real_density = True

        plt.show()

    def calculate_numpy_array_of_delta_x(self):

        delta_x_storage = np.zeros(
            [self.number_of_states, self.number_of_past_samples_considered_for_kde])
        for i, delta_x in enumerate(self.delta_x_deque):
            delta_x_storage[:, i] = delta_x

        return delta_x_storage

    def get_kde_independent_dist(self,k):
        """This function calculates the KDE for every state independently"""

        # This is used, if the disturbances on every state are not correlated with each other
        # Estimating a univariate KDE for every state needs less samples than estimating a multivariate KDE for a joint distribution

        # Determine weights for each kde
        self.determine_weights_for_indep(k)

        # Store and later return KDE for every state
        # Those KDEs are later used in the constraint tightening module
        kde_of_states = list()

        delta_x_storage = self.calculate_numpy_array_of_delta_x()

        for i in range(0, self.number_of_states):
            # A disturbance distribution is calculated for every state
            state_deviations = delta_x_storage[i, :]

            kde = stats.gaussian_kde(
                state_deviations, weights=self.weights[i,:])

            kde_of_states.append(kde)

        self.store_dist_x0(k)

        return kde_of_states

    def get_kde_multivariate_dist(self,k):
        """This function calculates a multivariate KDE"""

        # Calculating a multivariate KDE for all states takes longer to converge to the true density compared to calculating individual KDE for each state
        # Calculating a multivariate KDE is necessary if the random variables (disturbances of states) are correlated

        # Assumption, that in multivariate case, rate of change of disturbance distribution should be the same on each state
        self.determine_weights_for_indep(k)

        delta_x_storage = self.calculate_numpy_array_of_delta_x()

        kde = stats.gaussian_kde(
            delta_x_storage, weights=self.weights[0,:])

        self.store_dist_x0(k)

        return kde

    def determine_weights_for_indep(self,k):
        """This function determines the weights of the independent kde estimation using the Bhattacharyya coefficient
            First the recorded delta_x samples are divided in to groups

             1. Oldest 0.75*n samples
             2. Newest 0.25*n samples

            Then the kernel density estimation of these two sample set is calculated.
            Afterwards the Bhattacharyya coefficient of the two distributions is calculated
            A Bhattacharyya coefficient close to one indicates little change in the underlying distriubtion of the disturbance
            A Bhattacharyya coefficient close to zero indicates a big change in the underlying distriubtion of the disturbance

            The Bhattacharyya coefficient is then mapped on the interval 0.975-1.
            The result is used as a base to calculate the weights of the samples in the KDE used to calculate the disturbance distribution"""


        delta_x_storage = self.calculate_numpy_array_of_delta_x()
        idx_border_old_new = int(
            self.number_of_past_samples_considered_for_kde * 0.75)

        old_samples = delta_x_storage[:, :idx_border_old_new]
        new_samples = delta_x_storage[:, idx_border_old_new:]

        kde_old = list()
        kde_new = list()

        for i in range(0, self.number_of_states):
            kde_old.append(stats.gaussian_kde(old_samples[i, :]))
            kde_new.append(stats.gaussian_kde(new_samples[i, :]))

        # Eval Pdf
        interv_eval = np.linspace(self.interv_min, self.interv_max, self.number_eval_points)

        pdf_old = list()
        pdf_new = list()

        # Normalize pdf
        norm_factor = (self.interv_max-self.interv_min)/self.number_eval_points

        for i in range(0, self.number_of_states):
            pdf_old.append(kde_old[i].evaluate(interv_eval)*norm_factor)
            pdf_new.append(kde_new[i].evaluate(interv_eval)*norm_factor)

        # Calculate Bhattacharyya coefficient for each state
        b_coeff = list()

        for i in range(0, self.number_of_states):
            coeff = pdf_old[i]*pdf_new[i]
            coeff = np.sqrt(coeff)
            coeff = np.sum(coeff)
            b_coeff.append(coeff)

        if self.record_data:
            filename_b_coeff = os.path.join(self.b_coeff_path,"b_coeff_k_"+str(k))
            np.save(filename_b_coeff,np.array(b_coeff))

        print(f"Bhattacharyya coefficient: {b_coeff}")

        # Calculate base for weights
        base_weights = np.zeros([self.number_of_states])

        for i in range(0, self.number_of_states):
            base = 0.975 + 0.025 * b_coeff[i]
            base_weights[i] = base

        # Calculate weights
        self.weights = np.zeros(delta_x_storage.shape)

        for i in range(0, self.number_of_past_samples_considered_for_kde):
            cur_exponent = self.number_of_past_samples_considered_for_kde-i
            self.weights[:, i] = np.power(base_weights, cur_exponent)

        if self.record_data:
            filename_weights = os.path.join(self.weights_path,"weights_k_"+str(k))
            np.save(filename_weights,np.array(self.weights))

    def store_dist_x0(self,k):
        if self.record_data:
            interval = np.linspace(self.interv_min,self.interv_max,self.number_eval_points)

            delta_x_storage = self.calculate_numpy_array_of_delta_x()
            kde = stats.gaussian_kde(delta_x_storage[0,:],weights=self.weights[0,:])

            # Evaluate pdf and normalize result
            pdf = kde.evaluate(interval)*(self.interv_max-self.interv_min)/self.number_eval_points

            filename_estim_pdf = os.path.join(self.dist_x0_path,"estim_pdf_k_"+str(k))
            np.save(filename_estim_pdf,pdf)
