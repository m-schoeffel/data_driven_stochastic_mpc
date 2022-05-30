import numpy as np

# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF


class GaussianProcess:
    def __init__(self):

        # Store index k and delta x (difference between prediction and actual state)
        self.array_delta_x = np.ones((100,2))
        self.numbr_measurements=0

    def add_delta_x(self, index_k, delta_x):
        self.array_delta_x[self.numbr_measurements,:] = ([index_k, delta_x])
        self.numbr_measurements += 1

    def plot_distribution(self):
        # Todo: Distribution should be plottet here
        # Right now self.delta will be displayed

        # print(self.list_delta_x)
        print(self.array_delta_x[0:self.numbr_measurements, 0])
        print(self.array_delta_x[0:self.numbr_measurements+1, 1])

        # kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        # gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        # gaussian_process.fit(self.list_delta_x[0,:], self.list_delta_x[1,:])
        # gaussian_process.kernel_
