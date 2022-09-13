import time
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.metrics import pairwise_distances_chunked

from lti_system import lti_system
from data_driven_mpc import data_driven_predictor
from data_driven_mpc import data_driven_mpc
from lti_system import disturbance
from config import load_parameters
from disturbance_estimation import gaussian_process, traditional_kernel_density_estimator, discounted_kernel_density_estimator

from . import create_modules


[main_param, lti_system_param] = load_parameters.load_parameters()

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
    
    # # ------------------ Plot state sequence ------------------
    # fig,axs = plt.subplots(2,2)    
    # x_values = list(range(0,NUMBER_OF_MEASUREMENTS))
    # axs[0,0].plot(x_values,state_storage[0,:],label="x_0")
    # axs[0,1].plot(x_values,state_storage[1,:],label="x_1")
    # axs[1,0].plot(x_values,state_storage[2,:],label="x_2")
    # axs[1,1].plot(x_values,state_storage[3,:],label="x_3")

    # plt.figure()
    # plt.plot(state_storage[0,:],state_storage[1,:])

    # plt.show()



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





    # ------------------ Plot disturbance ------------------
    # print(disturbance_estimator.delta_x_array.shape)

    # plot_real_density, fig, ax = disturbance_estimator.plot_distribution()

    # # Only put real density in plot when it makes sense (not for gaussian process)
    # if plot_real_density:
    #     for i, dist_type in enumerate(TYPES_OF_DISTURBANCES):
    #         print(dist_type)
    #         my_disturbance.plot_real_disturbance(ax[i],dist_type)

    # plt.show()


if __name__ == "__main__":
    main()
