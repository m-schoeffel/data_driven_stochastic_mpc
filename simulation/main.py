import time
import numpy as np

from config import load_parameters
from graphics import animate_state_sequence, plot_state_sequence, plot_disturbance

from . import create_modules




def main():
    main_param = load_parameters.load_main_params()
    NUMBER_OF_MEASUREMENTS = main_param["number_of_measurements"]

    lti_system_param = load_parameters.load_lti_system_params()
    X_INITIAL_STATE = lti_system_param["x_0"]

    real_system = create_modules.create_system()

    [dd_mpc, dd_predictor, disturbance_estimator] = create_modules.create_controller_modules(real_system)

    # Set initial state
    real_system.x = X_INITIAL_STATE
    print(f"initial state: \n{real_system.x}")

    state_storage = np.zeros([X_INITIAL_STATE.shape[0],NUMBER_OF_MEASUREMENTS])
    
    print(f"ms1: {NUMBER_OF_MEASUREMENTS}")
    for i in range(0, NUMBER_OF_MEASUREMENTS):
        start_time = time.time()
        # print(f"\n\nk = {real_system.k}:")

        # Todo: Nächste Zeile muss mit MPC ausgetauscht werden
        u = np.random.randint(-5,5,size=(1,2))
        next_u = dd_mpc.get_new_u(real_system.x,goal_state=[-2,-2,0,0])
        # print(next_u)

        predicted_state = dd_predictor.predict_state(real_system.x, next_u)

        # print(f"Predicted state:  {predicted_state}")
        real_system.next_step(next_u,add_disturbance=False)
        # print(f"actual state: \n{real_system.x}")

        delta_x = real_system.x - predicted_state

        state_storage[:,i]=real_system.x.reshape(-1)

        disturbance_estimator.add_delta_x(real_system.k, delta_x)
        print("--- \"Main Loop\" took %s seconds ---" % (time.time() - start_time))
    




    animate_state_sequence.animate_state_sequence(state_storage)
    # plot_state_sequence.plot_state_sequence(state_storage,NUMBER_OF_MEASUREMENTS)
    # plot_disturbance.plot_disturbance_estimation(disturbance_estimator)

if __name__ == "__main__":
    main()
