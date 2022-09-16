import time
import numpy as np

from config import load_parameters
from graphics import animate_state_sequence, plot_state_sequence

from simulation import create_modules


def main():
    main_param = load_parameters.load_main_params()
    number_of_measurements = main_param["number_of_measurements"]

    lti_system_param = load_parameters.load_lti_system_params()
    x_initial_state = lti_system_param["x_0"]

    real_system = create_modules.create_system()

    [dd_mpc, dd_predictor,
        disturbance_estimator, constraint_tightener] = create_modules.create_controller_modules(real_system)

    # Set initial state
    real_system.x = x_initial_state
    print(f"initial state: \n{real_system.x}")

    state_storage = np.zeros(
        [x_initial_state.shape[0], number_of_measurements])
    g_z_storage = list()

    for i in range(0, number_of_measurements):
        start_time = time.time()

        dist_intervals = disturbance_estimator.get_disturbance_intervals()
        [G_v, g_v, G_z, g_z] = constraint_tightener.tighten_constraints_on_interv(dist_intervals)
        next_u = dd_mpc.get_new_u(
            real_system.x, G_v, g_v, G_z, g_z, goal_state=[-2, -2, 0, 0])
        predicted_state = dd_predictor.predict_state(real_system.x, next_u)
        real_system.next_step(next_u, add_disturbance=True)

        # Save data (states, tightened_constraints, etc.) for animation
        state_storage[:, i] = real_system.x.reshape(-1)
        g_z_storage.append(g_z)


        delta_x = real_system.x - predicted_state
        disturbance_estimator.add_delta_x(real_system.k, delta_x)

        print("--- \"Main Loop\" took %s seconds ---" %
             (time.time() - start_time))

    disturbance_estimator.get_disturbance_intervals()

    # animate_state_sequence.animate_state_sequence(state_storage)
    # plot_state_sequence.plot_state_sequence(state_storage,number_of_measurements)
    disturbance_estimator.plot_distribution()


if __name__ == "__main__":
    main()
