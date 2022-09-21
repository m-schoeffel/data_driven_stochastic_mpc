import time
import numpy as np

from config import load_parameters
from graphics import animate_state_sequence, plot_state_sequence

from simulation import create_modules


def main():
    main_param = load_parameters.load_main_params()
    prediction_horizon = load_parameters.load_prediction_horizon()


    # Currently a reference is being tracked, so the number_of_measurements corresponds to the number of reference samples
    number_of_measurements = main_param["number_of_measurements"]

    lti_system_param = load_parameters.load_lti_system_params()
    x_initial_state = lti_system_param["x_0"]

    real_system = create_modules.create_system()

    [dd_mpc,
        disturbance_estimator, constraint_tightener] = create_modules.create_controller_modules(real_system)

    # Load reference trajectory
    ref_traj = np.genfromtxt(
        "lti_system/reference_trajectories/"+"ref_traj_1.csv", delimiter=',')
    print(ref_traj[:, 25].reshape(-1))

    # number_of_measurements = ref_traj.shape[1]

    # Set initial state
    real_system.x = x_initial_state
    real_system.x[:] = ref_traj[:, 0].reshape(-1, 1)
    print(f"initial state: \n{real_system.x}")

    state_storage = np.zeros(
        [x_initial_state.shape[0], number_of_measurements])
    g_z_storage = list()

    for i in range(0, number_of_measurements):
        start_time = time.time()

        dist_intervals = disturbance_estimator.get_disturbance_intervals()
        [G_v, g_v, G_z, g_z] = constraint_tightener.tighten_constraints_on_interv(
            dist_intervals)
        ref_pred_hor = ref_traj[:, i:i+prediction_horizon]
        [next_u, x_pred] = dd_mpc.get_new_u(
            real_system.x, G_v, g_v, G_z, g_z, ref_pred_hor=ref_pred_hor)
        real_system.next_step(next_u, add_disturbance=True)

        # Save data (states, tightened_constraints, etc.) for animation
        state_storage[:, i] = real_system.x.reshape(-1)
        g_z_storage.append(g_z.copy())

        delta_x = real_system.x - x_pred
        disturbance_estimator.add_delta_x(real_system.k, delta_x)

        print(f"goal state x: {ref_traj[0,i]}")
        print(f"goal state y: {ref_traj[1,i]}")
        print(f"current state x: {real_system.x[0]}")
        print(f"current state y: {real_system.x[1]}")
        print(f"pseudo constraint x1: {g_z[0]}")

        print("--- \"Main Loop\" took %s seconds ---" %
              (time.time() - start_time))
        print()

    animate_state_sequence.animate_state_sequence(state_storage,g_z_storage,ref_traj)
    # plot_state_sequence.plot_state_sequence(state_storage,number_of_measurements)
    disturbance_estimator.plot_distribution()


if __name__ == "__main__":
    main()
