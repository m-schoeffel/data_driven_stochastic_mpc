import warnings

import numpy as np
import matplotlib.pyplot as plt


class Disturbance:
    def __init__(self, types_of_disturbances: list()):
        self.types_of_disturbances = types_of_disturbances

    def get_specific_dist(self, type_of_disturbance) -> float:
        if type_of_disturbance == "gaussian":
            return np.random.normal(loc=0, scale=0.1)
        elif type_of_disturbance == "uniform":
            return np.random.uniform(low=-0.1, high=0.1)
        elif type_of_disturbance == "triangular":
            return np.random.triangular(left=-2, mode=0.5, right=1)
        elif type_of_disturbance == "lognormal":
            return np.random.lognormal(mean=0.0, sigma=1.0)

        warnings.warn(
            "No proper disturbance specified, check global variable TYPE_OF_DISTURBANCE")
        return 0

    def get_dist_vector(self) -> list():
        dist_list = [self.get_specific_dist(
            dist_type) for dist_type in self.types_of_disturbances]
        dist_vector = np.array(dist_list).reshape(-1, 1)
        return dist_vector

    def plot_real_disturbance(self, ax, type_of_disturbance) -> None:
        number_samples = 1000
        x = np.linspace(-5, 5, number_samples).reshape(-1, 1)

        if type_of_disturbance == "gaussian":
            mu = 0
            sigma = 1.0
            pdf = 1/(sigma * np.sqrt(2 * np.pi)) * \
                np.exp(- (x - mu)**2 / (2 * sigma**2))
        elif type_of_disturbance == "uniform":
            lower_bound = -0.5
            upper_bound = 0.5
            x = np.linspace(lower_bound, upper_bound, number_samples)
            pdf = [1/(upper_bound-lower_bound) for _ in x]
        elif type_of_disturbance == "triangular":
            left = -2
            mode = 0.5
            right = 1
            x = np.linspace(left, right, number_samples)
            def left_side(x): return 2*(x-left)/((right-left)*(mode-left))
            def right_side(x): return 2*(right-x)/((right-left)*(right-mode))
            pdf = [left_side(x)
                   for x in x if x >= left and x <= mode]
            pdf = pdf + [right_side(x)
                         for x in x if x > mode and x <= right]
        elif type_of_disturbance == "lognormal":
            mu = 0
            sigma = 1.0
            pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) /
                   (x * sigma * np.sqrt(2 * np.pi)))
        else:
            return
        ax.plot(x, pdf, linewidth=2, color='r')