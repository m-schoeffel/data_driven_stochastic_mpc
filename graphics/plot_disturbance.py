import matplotlib.pyplot as plt


def plot_disturbance_estimation(disturbance_estimator):
    # ------------------ Plot disturbance ------------------
    print(disturbance_estimator.delta_x_array.shape)

    plot_real_density, fig, ax = disturbance_estimator.plot_distribution()

    plt.show()