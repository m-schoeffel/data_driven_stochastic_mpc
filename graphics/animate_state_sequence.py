import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from config import load_parameters

# ------------------ Animate state sequence ------------------

# The animation is specific to the current system and has to be adapted to a new system


def animate_state_sequence(state_storage,g_z_storage,ref_traj):
    fig = plt.figure()
    ax = plt.axes(xlim=(-4, 4), ylim=(-4, 4))

    len_traj = state_storage.shape[1]
    timesteps = list(range(0,len_traj))

    # prepare plots for joint constraints for x1 and x2
    # Todo: Make plottet constraints truly flexible
    constraints = load_parameters.load_constraints()

    G_x = np.array(constraints["G_x"])
    g_x = np.array(constraints["g_x"])
    x1_constr = np.ones(len_traj)*g_x[0]


    def animate(k):
        # Plot system state x_1 at time k
        ax.scatter(k, state_storage[0, k], color='red')

        # Plot real constraint
        ax.plot(timesteps, x1_constr, color='blue', lw=3)

        # Plot tightened pseudo constraints
        curr_tight_x1_constr = g_z_storage[0][0]
        curr_tight_x1_constr_array = np.ones(len_traj)*curr_tight_x1_constr
        ax.plot(timesteps,curr_tight_x1_constr_array,color='green',lw=3)

        # Plot reference trajectory
        ax.plot(timesteps,ref_traj[0,0:len_traj])

        ax.set_xlim(0,len_traj)

    anim = animation.FuncAnimation(fig, animate, interval=10)

    plt.show()
