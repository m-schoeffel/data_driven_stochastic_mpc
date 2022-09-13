import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from config import load_parameters

from . import create_modules


lti_system_param = load_parameters.load_lti_system_params()
main_param = load_parameters.load_main_params()

NUMBER_OF_MEASUREMENTS = main_param["number_of_measurements"]

X_INITIAL_STATE = lti_system_param["x_0"]

def main():

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
    




    # ------------------ Animate state sequence ------------------
    
    fig = plt.figure()
    ax = plt.axes(xlim=(-4, 4), ylim=(-4, 4))

    # prepare plots for joint constraints for x1 and x2
    # Todo: Make plottet constraints truly flexible
    constraints = load_parameters.load_constraints()
    G_x = np.array(constraints["G_x"])
    g_x = np.array(constraints["g_x"])
    x1_constr=np.ones(100)*g_x[0]
    y1_constr=np.linspace(-4,4,100)

    x2_constr=np.linspace(-4,4,100)
    y2_constr=np.ones(100)*g_x[1]


    def animate(i):
        ax.scatter(state_storage[0,i],state_storage[1,i])
        ax.scatter(0,0)

        ax.plot(x1_constr,y1_constr,color='blue',lw=10)
        ax.plot(x2_constr,y2_constr,color='blue',lw=10)

    anim = animation.FuncAnimation(fig, animate, interval=1000)

    plt.show()

if __name__ == "__main__":
    main()
