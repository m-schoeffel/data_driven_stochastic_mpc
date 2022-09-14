import matplotlib.pyplot as plt


def plot_state_sequence(state_storage, number_of_measurements):
    # ------------------ Plot state sequence ------------------
    fig, axs = plt.subplots(2, 2)
    x_values = list(range(0, number_of_measurements))
    axs[0, 0].plot(x_values, state_storage[0, :], label="x_0")
    axs[0, 1].plot(x_values, state_storage[1, :], label="x_1")
    axs[1, 0].plot(x_values, state_storage[2, :], label="x_2")
    axs[1, 1].plot(x_values, state_storage[3, :], label="x_3")

    plt.figure()
    plt.plot(state_storage[0, :], state_storage[1, :])

    plt.show()
